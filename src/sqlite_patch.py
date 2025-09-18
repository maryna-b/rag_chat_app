"""
SQLite3 compatibility patch for ChromaDB on Streamlit Cloud.

This module ensures ChromaDB works with older SQLite3 versions by
substituting pysqlite3 when needed.
"""

def patch_sqlite():
    """Patch SQLite3 for ChromaDB compatibility on deployment platforms."""
    import sys

    # Only apply patch in deployment environments (not locally)
    # Check if we're likely in Streamlit Cloud
    is_streamlit_cloud = (
        'streamlit' in sys.modules or
        '/mount/src/' in sys.path[0] if sys.path else False
    )

    if is_streamlit_cloud:
        try:
            # Try to import pysqlite3 and replace sqlite3
            import pysqlite3 as sqlite3
            sys.modules['sqlite3'] = sqlite3
            print("✅ Applied SQLite3 patch for ChromaDB compatibility")
        except ImportError:
            print("⚠️  pysqlite3-binary not found, using system SQLite3")
            pass

    return True