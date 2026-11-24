import * as crypto from 'crypto';
import { QdrantClient } from '@qdrant/js-client-rest';
import {
    VectorDocument,
    SearchOptions,
    VectorSearchResult,
    VectorDatabase,
    HybridSearchRequest,
    HybridSearchOptions,
    HybridSearchResult,
} from './types';

export interface QdrantConfig {
    url: string;
    apiKey?: string;
}

const DENSE = 'vector';
const SPARSE = 'sparse_vector';
const SCROLL_PAGE = 1000;
// BM25 term-frequency saturation; Qdrant applies the IDF half via the sparse "Idf" modifier.
const BM25_K1 = 1.2;

// Qdrant point IDs must be UUIDs or integers, so string chunk IDs are mapped to a stable UUID.
// The original ID is kept in the payload as `id`.
function toPointId(id: string): string {
    const h = crypto.createHash('md5').update(id).digest('hex');
    return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20, 32)}`;
}

// Splits identifiers too (camelCase, snake_case) so `getUserName` matches "user name".
export function tokenize(text: string): string[] {
    return text
        .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
        .toLowerCase()
        .split(/[^a-z0-9]+/)
        .filter(t => t.length > 1);
}

function termIndex(term: string): number {
    return crypto.createHash('md5').update(term).digest().readUInt32BE(0);
}

export function toSparseVector(text: string, isQuery: boolean): { indices: number[]; values: number[] } {
    const tf = new Map<number, number>();
    for (const t of tokenize(text)) {
        const i = termIndex(t);
        tf.set(i, (tf.get(i) || 0) + 1);
    }
    const indices = [...tf.keys()];
    const values = indices.map(i => isQuery ? 1 : (tf.get(i)! * (BM25_K1 + 1)) / (tf.get(i)! + BM25_K1));
    return { indices, values };
}

// claude-context only builds two filter shapes; anything else is rejected rather than silently ignored.
export function toQdrantFilter(expr?: string): Record<string, any> | undefined {
    if (!expr || !expr.trim()) return undefined;
    const eq = expr.match(/^\s*(\w+)\s*==\s*"((?:[^"\\]|\\.)*)"\s*$/);
    if (eq) {
        return { must: [{ key: eq[1], match: { value: eq[2].replace(/\\(.)/g, '$1') } }] };
    }
    const inList = expr.match(/^\s*(\w+)\s+in\s+\[(.*)\]\s*$/);
    if (inList) {
        const values = [...inList[2].matchAll(/"((?:[^"\\]|\\.)*)"/g)].map(m => m[1].replace(/\\(.)/g, '$1'));
        return { must: [{ key: inList[1], match: { any: values } }] };
    }
    throw new Error(`Unsupported filter expression for Qdrant: ${expr}`);
}

function toDocument(payload: Record<string, any> | null | undefined): VectorDocument {
    const p = payload || {};
    return {
        id: p.id,
        vector: [],
        content: p.content,
        relativePath: p.relativePath,
        startLine: p.startLine,
        endLine: p.endLine,
        fileExtension: p.fileExtension,
        metadata: JSON.parse(p.metadata || '{}'),
    };
}

export class QdrantVectorDatabase implements VectorDatabase {
    private client: QdrantClient;

    constructor(config: QdrantConfig) {
        const url = new URL(config.url);
        // The client defaults to port 6333 when the URL has none, which breaks https URLs on 443.
        const port = url.port ? Number(url.port) : (url.protocol === 'https:' ? 443 : 6333);
        this.client = new QdrantClient({ url: config.url, port, apiKey: config.apiKey, checkCompatibility: false });
    }

    async createCollection(collectionName: string, dimension: number, description?: string): Promise<void> {
        await this.client.createCollection(collectionName, {
            vectors: { [DENSE]: { size: dimension, distance: 'Cosine' } },
            metadata: { description: description || '' },
        });
        await this.createPayloadIndexes(collectionName);
    }

    async createHybridCollection(collectionName: string, dimension: number, description?: string): Promise<void> {
        await this.client.createCollection(collectionName, {
            vectors: { [DENSE]: { size: dimension, distance: 'Cosine' } },
            sparse_vectors: { [SPARSE]: { modifier: 'idf' } },
            metadata: { description: description || '' },
        });
        await this.createPayloadIndexes(collectionName);
    }

    private async createPayloadIndexes(collectionName: string): Promise<void> {
        for (const field of ['relativePath', 'fileExtension']) {
            await this.client.createPayloadIndex(collectionName, { field_name: field, field_schema: 'keyword', wait: true });
        }
    }

    async dropCollection(collectionName: string): Promise<void> {
        await this.client.deleteCollection(collectionName);
    }

    async hasCollection(collectionName: string): Promise<boolean> {
        return (await this.client.collectionExists(collectionName)).exists;
    }

    async listCollections(): Promise<string[]> {
        return (await this.client.getCollections()).collections.map(c => c.name);
    }

    async insert(collectionName: string, documents: VectorDocument[]): Promise<void> {
        await this.upsert(collectionName, documents, false);
    }

    async insertHybrid(collectionName: string, documents: VectorDocument[]): Promise<void> {
        await this.upsert(collectionName, documents, true);
    }

    private async upsert(collectionName: string, documents: VectorDocument[], hybrid: boolean): Promise<void> {
        if (documents.length === 0) return;
        await this.client.upsert(collectionName, {
            wait: true,
            points: documents.map(doc => ({
                id: toPointId(doc.id),
                vector: hybrid
                    ? { [DENSE]: doc.vector, [SPARSE]: toSparseVector(doc.content, false) }
                    : { [DENSE]: doc.vector },
                payload: {
                    id: doc.id,
                    content: doc.content,
                    relativePath: doc.relativePath,
                    startLine: doc.startLine,
                    endLine: doc.endLine,
                    fileExtension: doc.fileExtension,
                    metadata: JSON.stringify(doc.metadata),
                },
            })),
        });
    }

    async search(collectionName: string, queryVector: number[], options?: SearchOptions): Promise<VectorSearchResult[]> {
        const res = await this.client.query(collectionName, {
            query: queryVector,
            using: DENSE,
            limit: options?.topK || 10,
            filter: toQdrantFilter(options?.filterExpr),
            score_threshold: options?.threshold,
            with_payload: true,
        });
        return res.points.map(p => ({ document: toDocument(p.payload), score: p.score }));
    }

    async hybridSearch(collectionName: string, searchRequests: HybridSearchRequest[], options?: HybridSearchOptions): Promise<HybridSearchResult[]> {
        const filter = toQdrantFilter(options?.filterExpr);
        const prefetch = searchRequests.map(req => ({
            query: typeof req.data === 'string' ? toSparseVector(req.data, true) : req.data,
            using: req.anns_field,
            limit: req.limit,
            filter,
        }));
        const res = await this.client.query(collectionName, {
            prefetch,
            query: { fusion: 'rrf' },
            limit: options?.limit || 10,
            filter,
            with_payload: true,
        });
        return res.points.map(p => ({ document: toDocument(p.payload), score: p.score }));
    }

    async delete(collectionName: string, ids: string[]): Promise<void> {
        if (ids.length === 0) return;
        await this.client.delete(collectionName, { wait: true, points: ids.map(toPointId) });
    }

    async query(collectionName: string, filter: string, outputFields: string[], limit?: number): Promise<Record<string, any>[]> {
        const results: Record<string, any>[] = [];
        let offset: any = undefined;
        do {
            const page = await this.client.scroll(collectionName, {
                filter: toQdrantFilter(filter),
                limit: limit ? Math.min(limit - results.length, SCROLL_PAGE) : SCROLL_PAGE,
                offset,
                with_payload: true,
                with_vector: false,
            });
            for (const p of page.points) {
                const payload = p.payload || {};
                results.push(Object.fromEntries(outputFields.map(f => [f, payload[f]])));
            }
            offset = page.next_page_offset;
        } while (offset !== undefined && offset !== null && (!limit || results.length < limit));
        return results;
    }

    async getCollectionDescription(collectionName: string): Promise<string> {
        const info: any = await this.client.getCollection(collectionName);
        return info.config?.metadata?.description || '';
    }

    async checkCollectionLimit(): Promise<boolean> {
        return true; // self-hosted Qdrant has no collection quota
    }

    async getCollectionRowCount(collectionName: string): Promise<number> {
        try {
            return (await this.client.count(collectionName, { exact: true })).count;
        } catch {
            return -1;
        }
    }
}
