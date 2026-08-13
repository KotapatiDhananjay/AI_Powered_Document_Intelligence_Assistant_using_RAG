/**
 * AI Document Intelligence Assistant — Frontend Application
 * 
 * Shared utilities and API client used across all pages.
 * Provides: API, Auth, Toast, markdown rendering, and helper functions.
 */

// ============================================================
//  API Base URL — auto-detect from current origin
// ============================================================
const API_BASE = '';

// ============================================================
//  API Client — Authenticated HTTP wrapper
// ============================================================
const API = {
    /**
     * Make an authenticated GET request.
     * @param {string} path - API endpoint path (e.g., '/api/documents')
     * @returns {Promise<any>} Parsed JSON response
     */
    async get(path) {
        const res = await fetch(`${API_BASE}${path}`, {
            method: 'GET',
            headers: API._headers(),
        });
        return API._handleResponse(res);
    },

    /**
     * Make an authenticated POST request with JSON body.
     * @param {string} path - API endpoint path
     * @param {object} body - Request body
     * @returns {Promise<any>} Parsed JSON response
     */
    async post(path, body) {
        const res = await fetch(`${API_BASE}${path}`, {
            method: 'POST',
            headers: { ...API._headers(), 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        return API._handleResponse(res);
    },

    /**
     * Make an authenticated DELETE request.
     * @param {string} path - API endpoint path
     * @returns {Promise<any>} Parsed JSON response or null for 204
     */
    async delete(path) {
        const res = await fetch(`${API_BASE}${path}`, {
            method: 'DELETE',
            headers: API._headers(),
        });
        if (res.status === 204) return null;
        return API._handleResponse(res);
    },

    /**
     * Upload a file with multipart/form-data.
     * @param {string} path - API endpoint path
     * @param {File} file - File object to upload
     * @returns {Promise<any>} Parsed JSON response
     */
    async upload(path, file) {
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch(`${API_BASE}${path}`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${Auth.getToken()}`,
            },
            body: formData,
        });
        return API._handleResponse(res);
    },

    /**
     * Build standard headers with JWT auth.
     * @returns {object} Headers object
     */
    _headers() {
        const headers = {};
        const token = Auth.getToken();
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        return headers;
    },

    /**
     * Handle HTTP response — parse JSON, throw on error.
     * @param {Response} res - Fetch Response object
     * @returns {Promise<any>} Parsed response data
     */
    async _handleResponse(res) {
        if (res.status === 401) {
            // Token expired or invalid
            Auth.logout();
            window.location.href = 'login.html';
            throw new Error('Session expired. Please log in again.');
        }

        if (!res.ok) {
            let detail = `Request failed (${res.status})`;
            try {
                const errData = await res.json();
                detail = errData.detail || detail;
            } catch {
                // Could not parse error body
            }
            throw new Error(detail);
        }

        // Handle empty responses (e.g., 204 No Content)
        const contentType = res.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            return null;
        }

        return res.json();
    },
};


// ============================================================
//  Auth — JWT Token & User Management
// ============================================================
const Auth = {
    TOKEN_KEY: 'docai_token',
    USER_KEY: 'docai_user',

    /**
     * Register a new user.
     * @param {string} email
     * @param {string} username
     * @param {string} password
     * @returns {Promise<object>} Token response with user data
     */
    async register(email, username, password) {
        const data = await API.post('/api/auth/register', { email, username, password });
        Auth._save(data);
        return data;
    },

    /**
     * Log in an existing user.
     * @param {string} email
     * @param {string} password
     * @returns {Promise<object>} Token response with user data
     */
    async login(email, password) {
        const data = await API.post('/api/auth/login', { email, password });
        Auth._save(data);
        return data;
    },

    /**
     * Log out — clear all stored auth data.
     */
    logout() {
        localStorage.removeItem(Auth.TOKEN_KEY);
        localStorage.removeItem(Auth.USER_KEY);
    },

    /**
     * Check if user is authenticated (has a token).
     * @returns {boolean}
     */
    isAuthenticated() {
        return !!Auth.getToken();
    },

    /**
     * Get the stored JWT token.
     * @returns {string|null}
     */
    getToken() {
        return localStorage.getItem(Auth.TOKEN_KEY);
    },

    /**
     * Get the stored user object.
     * @returns {object|null}
     */
    getUser() {
        try {
            const raw = localStorage.getItem(Auth.USER_KEY);
            return raw ? JSON.parse(raw) : null;
        } catch {
            return null;
        }
    },

    /**
     * Save token and user to localStorage.
     * @param {object} data - Token response from API
     */
    _save(data) {
        if (data.access_token) {
            localStorage.setItem(Auth.TOKEN_KEY, data.access_token);
        }
        if (data.user) {
            localStorage.setItem(Auth.USER_KEY, JSON.stringify(data.user));
        }
    },
};


// ============================================================
//  Toast Notifications
// ============================================================
const Toast = {
    /**
     * Show a toast notification.
     * @param {string} message - Notification text
     * @param {'success'|'error'|'info'|'warning'} type - Toast type
     * @param {number} duration - Auto-dismiss duration in ms (default 5000)
     */
    show(message, type = 'info', duration = 5000) {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const icons = {
            success: '✅',
            error: '❌',
            info: 'ℹ️',
            warning: '⚠️',
        };

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <span class="toast-icon">${icons[type] || 'ℹ️'}</span>
            <span>${escapeHtml(message)}</span>
            <button class="toast-close" onclick="this.parentElement.remove()">✕</button>
        `;

        container.appendChild(toast);

        // Auto-dismiss
        setTimeout(() => {
            if (toast.parentElement) {
                toast.style.opacity = '0';
                toast.style.transform = 'translateX(100px)';
                toast.style.transition = 'all 0.3s ease';
                setTimeout(() => toast.remove(), 300);
            }
        }, duration);
    },
};


// ============================================================
//  Markdown Renderer — Lightweight, no dependencies
// ============================================================

/**
 * Render a markdown string to HTML.
 * Supports: headers, bold, italic, inline code, code blocks,
 * ordered/unordered lists, line breaks, and paragraphs.
 * 
 * @param {string} text - Raw markdown text
 * @returns {string} HTML string
 */
function renderMarkdown(text) {
    if (!text) return '';

    let html = escapeHtml(text);

    // Code blocks (```)
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Bold + Italic
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Unordered lists (- or •)
    html = html.replace(/^[-•] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

    // Ordered lists
    html = html.replace(/^\d+\.\s(.+)$/gm, '<li>$1</li>');
    // Wrap consecutive <li> that follow ordered pattern in <ol>
    // (simplified — wraps all remaining unwrapped <li> in <ol>)
    html = html.replace(/(?:^|(?<=<\/ul>))(<li>.*<\/li>\n?)+/gm, (match) => {
        if (!match.includes('<ul>')) {
            return `<ol>${match}</ol>`;
        }
        return match;
    });

    // Line breaks → paragraphs
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');

    // Wrap in paragraph if not already wrapped in a block element
    if (!html.startsWith('<h') && !html.startsWith('<pre') && !html.startsWith('<ul') && !html.startsWith('<ol')) {
        html = `<p>${html}</p>`;
    }

    // Clean up empty paragraphs
    html = html.replace(/<p><\/p>/g, '');
    html = html.replace(/<p><br><\/p>/g, '');

    return html;
}


// ============================================================
//  Helper Functions
// ============================================================

/**
 * Escape HTML special characters to prevent XSS.
 * @param {string} str - Raw string
 * @returns {string} Escaped HTML string
 */
function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/**
 * Get a file type icon emoji based on extension.
 * @param {string} fileType - File extension (without dot)
 * @returns {string} Emoji icon
 */
function getFileIcon(fileType) {
    const icons = {
        pdf: '📕',
        docx: '📘',
        doc: '📘',
        txt: '📄',
        pptx: '📙',
        ppt: '📙',
        csv: '📊',
        xlsx: '📊',
    };
    return icons[fileType?.toLowerCase()] || '📄';
}

/**
 * Format file size in human-readable format.
 * @param {number} bytes - File size in bytes
 * @returns {string} Formatted size string
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const k = 1024;
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + units[i];
}
