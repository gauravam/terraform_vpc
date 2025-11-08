SELECT DISTINCT 
    TRIM(SPLIT_PART(SPLIT_PART(a.plannode, ':', 2), ' ', 2)) AS table_name,
    COUNT(a.query) AS query_count
FROM stl_explain a
WHERE a.plannode LIKE '%missing statistics%'
  AND a.plannode NOT LIKE '%redshift_auto_health_check_%'
GROUP BY TRIM(SPLIT_PART(SPLIT_PART(a.plannode, ':', 2), ' ', 2))
ORDER BY query_count DESC;
