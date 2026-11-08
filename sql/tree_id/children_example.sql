.headers on
.mode column

-- Inspect children of a context through parent_id. Change ctx as needed.
WITH ctx AS (
  SELECT id
  FROM nodes
  WHERE direction = 0
    AND key = CAST('th' AS BLOB)
)
SELECT
  child.id,
  child.token,
  printf('%02X', child.token) AS hex,
  CASE
    WHEN child.token = 10 THEN '\\n'
    WHEN child.token = 13 THEN '\\r'
    WHEN child.token = 9 THEN '\\t'
    WHEN child.token BETWEEN 32 AND 126 THEN char(child.token)
    ELSE ''
  END AS text,
  child.mass
FROM nodes AS child, ctx
WHERE child.direction = 0
  AND child.parent_id = ctx.id
ORDER BY child.mass DESC, child.token
LIMIT 80;
