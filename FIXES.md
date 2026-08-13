# Fixes and Improvements

This document outlines the main fixes and UI improvements implemented in the DocIntel AI project.

## Backend & Configuration
- **SQLite Support**: Migrated the default database from PostgreSQL to SQLite (`sqlite+aiosqlite`) to simplify local setup and remove external dependencies.
- **Improved Error Handling**: Updated database initialization error catching to provide more accurate troubleshooting hints (especially for SQLite scenarios).
- **Hard-coded API URLs Removed**: Modified `app.js` to dynamically detect `API_BASE`, preventing localhost-specific bugs in production environments.
- **XLSX Document Support**: Integrated `openpyxl` in `loader.py` to extract text from Excel spreadsheets, chunking them efficiently into rows for the vector store.
- **Environment Configuration**: Added an `.env.example` file to guide new users in setting up API keys, JWT secrets, and database strings.
- **Conversation Sorting Fix**: Ensured the `Conversation` model's `updated_at` timestamp is explicitly updated whenever a new message is added, fixing the chronological ordering in the sidebar.

## Frontend & UI Improvements
- **SSE Streaming Chunk Loss**: Implemented a string buffer in `chat.html` to robustly handle Server-Sent Events (SSE) that split JSON payloads across network chunks.
- **Visual Theme Unified**: Updated the overall application color scheme to a consistent, premium deep space cyan/blue design.
- **Card Hover Tweaks**: Removed unnecessary and slightly jarring `translateY` scaling animations from `.glass-card:hover` states to improve the feel of the interface.
- **Accessibility Additions**: Implemented standard `:focus-visible` styling for robust keyboard navigation.
- **Motion Preferences**: Added CSS `@media (prefers-reduced-motion: reduce)` hooks to disable animations and transitions for users preferring reduced motion.

These changes make the application more robust, easier to run out of the box, and improve the user experience across all devices.
