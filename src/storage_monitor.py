
def exceeds_threshold(size_bytes, warning_mb):
    return size_bytes > warning_mb * 1024 * 1024


def storage_report(db, warning_mb=350):
    size = db.query('SELECT pg_database_size(current_database()) AS bytes, pg_size_pretty(pg_database_size(current_database())) AS pretty')[0]
    tables = db.query('''SELECT relname AS table_name, pg_total_relation_size(relid) AS bytes,
        pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
        pg_size_pretty(pg_indexes_size(relid)) AS index_size
        FROM pg_catalog.pg_statio_user_tables
        WHERE schemaname=current_schema() AND relname=ANY(%s)
        ORDER BY pg_total_relation_size(relid) DESC''',
        (['documents', 'document_chunks', 'memories', 'checkpoints', 'checkpoint_blobs', 'checkpoint_writes', 'app_threads', 'ingestion_records'],))
    indexes = db.query('''SELECT indexrelname AS index_name, pg_size_pretty(pg_relation_size(indexrelid)) AS size
        FROM pg_stat_user_indexes WHERE relname IN ('document_chunks','memories')''')
    counts = db.query('''SELECT (SELECT count(*) FROM app_threads) AS threads,
        (SELECT count(*) FROM checkpoints) AS checkpoints, (SELECT count(*) FROM documents) AS documents,
        (SELECT count(*) FROM document_chunks) AS chunks''')[0]
    memory_counts = db.query('SELECT memory_type, count(*) AS count FROM memories GROUP BY memory_type')
    return {'database': size, 'tables': tables, 'vector_indexes': indexes, 'counts': counts,
            'memories': memory_counts, 'warning': exceeds_threshold(size['bytes'], warning_mb)}
