Based on the utility functions for database migration analysis in the provided code, I'll create SQL commands that perform equivalent tests in Redshift using DBeaver. These commands will help analyze character sets, spaces, and VARCHAR compatibility for your columns.

Here are the SQL commands for each test:

## 1. Character Set Analysis

```sql
-- Character Set Analysis for a specific column
SELECT 
    '{your_column}' AS column_name,
    COUNT(*) AS total_rows,
    -- ASCII character analysis
    SUM(CASE WHEN "{your_column}" ~ '^[\x00-\x7F]*$' THEN 1 ELSE 0 END) AS ascii_only_rows,
    ROUND(100.0 * SUM(CASE WHEN "{your_column}" ~ '^[\x00-\x7F]*$' THEN 1 ELSE 0 END) / COUNT(*), 2) AS ascii_only_percent,
    
    -- Latin1 character analysis
    SUM(CASE WHEN "{your_column}" ~ '^[\x00-\xFF]*$' AND "{your_column}" !~ '^[\x00-\x7F]*$' THEN 1 ELSE 0 END) AS latin1_only_rows,
    ROUND(100.0 * SUM(CASE WHEN "{your_column}" ~ '^[\x00-\xFF]*$' AND "{your_column}" !~ '^[\x00-\x7F]*$' THEN 1 ELSE 0 END) / COUNT(*), 2) AS latin1_only_percent,
    
    -- UTF8 character analysis
    SUM(CASE WHEN "{your_column}" !~ '^[\x00-\xFF]*$' THEN 1 ELSE 0 END) AS utf8_only_rows,
    ROUND(100.0 * SUM(CASE WHEN "{your_column}" !~ '^[\x00-\xFF]*$' THEN 1 ELSE 0 END) / COUNT(*), 2) AS utf8_only_percent,
    
    -- Has non-ASCII content
    CASE WHEN SUM(CASE WHEN "{your_column}" !~ '^[\x00-\x7F]*$' THEN 1 ELSE 0 END) > 0 THEN 'Yes' ELSE 'No' END AS has_non_ascii,
    
    -- Has non-Latin1 content
    CASE WHEN SUM(CASE WHEN "{your_column}" !~ '^[\x00-\xFF]*$' THEN 1 ELSE 0 END) > 0 THEN 'Yes' ELSE 'No' END AS has_non_latin1
FROM 
    {your_schema}.{your_table}
WHERE 
    "{your_column}" IS NOT NULL;
```

## 2. Leading and Trailing Spaces Analysis

```sql
-- Space Analysis for a specific column
WITH space_analysis AS (
    SELECT 
        "{your_column}" AS value,
        LENGTH("{your_column}") AS total_length,
        LENGTH(LTRIM("{your_column}")) AS length_without_leading,
        LENGTH(RTRIM("{your_column}")) AS length_without_trailing,
        LENGTH(TRIM("{your_column}")) AS trimmed_length
    FROM 
        {your_schema}.{your_table}
    WHERE 
        "{your_column}" IS NOT NULL
)
SELECT 
    '{your_column}' AS column_name,
    COUNT(*) AS total_values,
    
    -- Leading spaces analysis
    SUM(CASE WHEN total_length > length_without_leading THEN 1 ELSE 0 END) AS values_with_leading_spaces,
    ROUND(100.0 * SUM(CASE WHEN total_length > length_without_leading THEN 1 ELSE 0 END) / COUNT(*), 2) AS percent_with_leading_spaces,
    
    -- Trailing spaces analysis
    SUM(CASE WHEN length_without_trailing < total_length THEN 1 ELSE 0 END) AS values_with_trailing_spaces,
    ROUND(100.0 * SUM(CASE WHEN length_without_trailing < total_length THEN 1 ELSE 0 END) / COUNT(*), 2) AS percent_with_trailing_spaces,
    
    -- Both leading and trailing
    SUM(CASE WHEN total_length > trimmed_length THEN 1 ELSE 0 END) AS values_with_any_spaces,
    ROUND(100.0 * SUM(CASE WHEN total_length > trimmed_length THEN 1 ELSE 0 END) / COUNT(*), 2) AS percent_with_any_spaces
FROM 
    space_analysis;
```

## 3. Sample Values with Spaces

```sql
-- Sample values with leading or trailing spaces
SELECT 
    CASE 
        WHEN LENGTH("{your_column}") > 23 
        THEN LEFT("{your_column}", 20) || '...' 
        ELSE "{your_column}" 
    END AS display_value,
    LENGTH("{your_column}") - LENGTH(LTRIM("{your_column}")) AS leading_spaces,
    LENGTH("{your_column}") - LENGTH(RTRIM("{your_column}")) AS trailing_spaces,
    '{your_column}' AS column_name
FROM 
    {your_schema}.{your_table}
WHERE 
    LENGTH("{your_column}") != LENGTH(TRIM("{your_column}"))
LIMIT 5;
```

## 4. VARCHAR Length Compatibility Analysis

```sql
-- VARCHAR size analysis for a column
WITH length_stats AS (
    SELECT 
        LENGTH("{your_column}") AS value_length
    FROM 
        {your_schema}.{your_table}
    WHERE 
        "{your_column}" IS NOT NULL
)
SELECT 
    '{your_column}' AS column_name,
    MAX(value_length) AS max_length,
    AVG(value_length) AS mean_length,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value_length) AS median_length,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY value_length) AS percentile_95th,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY value_length) AS percentile_99th,
    (SELECT COUNT(*) FROM {your_schema}.{your_table} WHERE "{your_column}" IS NULL) AS null_count,
    
    -- Redshift VARCHAR compatibility check (65535 character limit)
    CASE 
        WHEN MAX(value_length) <= 65535 THEN 'Compatible' 
        ELSE 'Exceeds Limit' 
    END AS redshift_compatibility,
    
    -- Recommended VARCHAR size with 20% buffer
    LEAST(CEIL(MAX(value_length) * 1.2), 65535) AS recommended_size
FROM 
    length_stats;
```

## 5. Distribution of Leading Space Counts

```sql
-- Distribution of leading space counts
WITH leading_space_counts AS (
    SELECT 
        LENGTH("{your_column}") - LENGTH(LTRIM("{your_column}")) AS leading_count
    FROM 
        {your_schema}.{your_table}
    WHERE 
        LENGTH("{your_column}") != LENGTH(LTRIM("{your_column}"))
        AND "{your_column}" IS NOT NULL
)
SELECT 
    leading_count,
    COUNT(*) AS frequency,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS percentage
FROM 
    leading_space_counts
GROUP BY 
    leading_count
ORDER BY 
    leading_count;
```

## 6. Distribution of Trailing Space Counts

```sql
-- Distribution of trailing space counts
WITH trailing_space_counts AS (
    SELECT 
        LENGTH("{your_column}") - LENGTH(RTRIM("{your_column}")) AS trailing_count
    FROM 
        {your_schema}.{your_table}
    WHERE 
        LENGTH("{your_column}") != LENGTH(RTRIM("{your_column}"))
        AND "{your_column}" IS NOT NULL
)
SELECT 
    trailing_count,
    COUNT(*) AS frequency,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS percentage
FROM 
    trailing_space_counts
GROUP BY 
    trailing_count
ORDER BY 
    trailing_count;
```

## 7. Comprehensive Column Analysis

```sql
-- Comprehensive column analysis combining all tests
WITH char_analysis AS (
    SELECT 
        SUM(CASE WHEN "{your_column}" ~ '^[\x00-\x7F]*$' THEN 1 ELSE 0 END) AS ascii_only_rows,
        SUM(CASE WHEN "{your_column}" ~ '^[\x00-\xFF]*$' AND "{your_column}" !~ '^[\x00-\x7F]*$' THEN 1 ELSE 0 END) AS latin1_only_rows,
        SUM(CASE WHEN "{your_column}" !~ '^[\x00-\xFF]*$' THEN 1 ELSE 0 END) AS utf8_only_rows,
        COUNT(*) AS total_rows
    FROM 
        {your_schema}.{your_table}
    WHERE 
        "{your_column}" IS NOT NULL
),
space_analysis AS (
    SELECT 
        SUM(CASE WHEN LENGTH("{your_column}") > LENGTH(LTRIM("{your_column}")) THEN 1 ELSE 0 END) AS with_leading_spaces,
        SUM(CASE WHEN LENGTH("{your_column}") > LENGTH(RTRIM("{your_column}")) THEN 1 ELSE 0 END) AS with_trailing_spaces,
        SUM(CASE WHEN LENGTH("{your_column}") > LENGTH(TRIM("{your_column}")) THEN 1 ELSE 0 END) AS with_any_spaces,
        COUNT(*) AS total_values
    FROM 
        {your_schema}.{your_table}
    WHERE 
        "{your_column}" IS NOT NULL
),
length_stats AS (
    SELECT 
        MAX(LENGTH("{your_column}")) AS max_length,
        AVG(LENGTH("{your_column}")) AS mean_length,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY LENGTH("{your_column}")) AS median_length,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY LENGTH("{your_column}")) AS percentile_95th,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY LENGTH("{your_column}")) AS percentile_99th
    FROM 
        {your_schema}.{your_table}
    WHERE 
        "{your_column}" IS NOT NULL
)
SELECT 
    '{your_column}' AS column_name,
    
    -- Character set stats
    ca.ascii_only_rows,
    ROUND(100.0 * ca.ascii_only_rows / ca.total_rows, 2) AS ascii_only_percent,
    ca.latin1_only_rows,
    ROUND(100.0 * ca.latin1_only_rows / ca.total_rows, 2) AS latin1_only_percent,
    ca.utf8_only_rows,
    ROUND(100.0 * ca.utf8_only_rows / ca.total_rows, 2) AS utf8_only_percent,
    CASE WHEN ca.latin1_only_rows > 0 OR ca.utf8_only_rows > 0 THEN 'Yes' ELSE 'No' END AS has_non_ascii,
    CASE WHEN ca.utf8_only_rows > 0 THEN 'Yes' ELSE 'No' END AS has_non_latin1,
    
    -- Space analysis
    sa.with_leading_spaces,
    ROUND(100.0 * sa.with_leading_spaces / sa.total_values, 2) AS percent_with_leading,
    sa.with_trailing_spaces,
    ROUND(100.0 * sa.with_trailing_spaces / sa.total_values, 2) AS percent_with_trailing,
    sa.with_any_spaces,
    ROUND(100.0 * sa.with_any_spaces / sa.total_values, 2) AS percent_with_any_spaces,
    
    -- Length analysis
    ls.max_length,
    ls.mean_length,
    ls.median_length,
    ls.percentile_95th,
    ls.percentile_99th,
    (SELECT COUNT(*) FROM {your_schema}.{your_table} WHERE "{your_column}" IS NULL) AS null_count,
    
    -- Redshift compatibility
    CASE WHEN ls.max_length <= 65535 THEN 'Compatible' ELSE 'Exceeds Limit' END AS redshift_compatibility,
    LEAST(CEIL(ls.max_length * 1.2), 65535) AS recommended_varchar_size,
    
    -- Encoding recommendation
    CASE 
        WHEN ca.utf8_only_rows > 0 THEN 'UTF8'
        WHEN ca.latin1_only_rows > 0 THEN 'LATIN1'
        ELSE 'ASCII'
    END AS recommended_encoding
FROM 
    char_analysis ca,
    space_analysis sa,
    length_stats ls;
```

For each of these SQL commands, replace:
- `{your_column}` with your actual column name
- `{your_schema}` with your schema name
- `{your_table}` with your table name

These SQL commands implement equivalent functionality to the Python functions in your code, helping you analyze character sets, spaces, and VARCHAR compatibility directly in Redshift using DBeaver.
