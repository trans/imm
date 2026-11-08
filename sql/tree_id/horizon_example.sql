.headers on
.mode column

-- Future-token distributions for a single context.
WITH ctx AS (
  SELECT id
  FROM nodes
  WHERE direction = 0
    AND key = CAST('th' AS BLOB)
)
SELECT
  hc.horizon,
  hc.target,
  printf('%02X', hc.target) AS hex,
  CASE
    WHEN hc.target = 10 THEN '\\n'
    WHEN hc.target = 13 THEN '\\r'
    WHEN hc.target = 9 THEN '\\t'
    WHEN hc.target BETWEEN 32 AND 126 THEN char(hc.target)
    ELSE ''
  END AS text,
  hc.mass,
  ROUND(CAST(hc.mass AS REAL) / SUM(hc.mass) OVER (PARTITION BY hc.horizon), 6) AS p
FROM horizon_counts hc, ctx
WHERE hc.node_id = ctx.id
ORDER BY hc.horizon, hc.mass DESC, hc.target;
