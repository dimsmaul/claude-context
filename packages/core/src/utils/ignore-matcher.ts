import ignore, { Ignore } from 'ignore';

export class IgnoreMatcher {
    private matcher: Ignore;

    constructor(patterns: string[] = []) {
        const cleanPatterns = patterns
            .map(pattern => pattern.trim())
            .filter(pattern => pattern.length > 0 && !pattern.startsWith('#'));

        this.matcher = ignore().add(cleanPatterns);
    }

    ignores(relativePath: string, isDirectory: boolean = false): boolean {
        const normalizedPath = this.normalizePath(relativePath);
        if (!normalizedPath) {
            return false;
        }

        if (this.hasHiddenSegment(normalizedPath)) {
            return true;
        }

        // Directories are matched only with a trailing slash, so `*` followed by `!*/`
        // (a common .dockerignore allowlist) does not skip every directory.
        if (isDirectory) {
            return this.matcher.ignores(`${normalizedPath}/`);
        }

        return this.matcher.ignores(normalizedPath);
    }

    private normalizePath(relativePath: string): string {
        return relativePath
            .replace(/\\/g, '/')
            .replace(/^\/+|\/+$/g, '');
    }

    private hasHiddenSegment(relativePath: string): boolean {
        return relativePath
            .split('/')
            .some(part => part.length > 0 && part.startsWith('.'));
    }
}
