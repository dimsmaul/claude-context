import { toQdrantFilter, toSparseVector, tokenize } from './qdrant-vectordb';

describe('Qdrant helpers', () => {
    it('translates the filter shapes Context builds', () => {
        expect(toQdrantFilter('')).toBeUndefined();
        expect(toQdrantFilter(undefined)).toBeUndefined();
        expect(toQdrantFilter('relativePath == "src/a \\"b\\".ts"')).toEqual({
            must: [{ key: 'relativePath', match: { value: 'src/a "b".ts' } }],
        });
        expect(toQdrantFilter('fileExtension in [".ts", ".go"]')).toEqual({
            must: [{ key: 'fileExtension', match: { any: ['.ts', '.go'] } }],
        });
        expect(() => toQdrantFilter('startLine > 3')).toThrow('Unsupported filter');
    });

    it('splits identifiers into terms', () => {
        expect(tokenize('getUserName(user_id)')).toEqual(['get', 'user', 'name', 'user', 'id']);
    });

    it('builds matching sparse terms for documents and queries', () => {
        const doc = toSparseVector('getUserName user', false);
        const query = toSparseVector('user name', true);
        expect(doc.indices).toHaveLength(3);
        expect(query.values).toEqual([1, 1]);
        for (const i of query.indices) expect(doc.indices).toContain(i);
        // repeated term weighs more but saturates
        const userWeight = doc.values[doc.indices.indexOf(query.indices[0])];
        expect(userWeight).toBeGreaterThan(1);
        expect(userWeight).toBeLessThan(2.2);
    });
});
