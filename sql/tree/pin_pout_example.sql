.headers on
.mode column

-- Compare outgoing distribution for a text key in the forward tree with
-- incoming distribution for the same text key in the reverse tree.
--
-- For ctx = 'th', reverse_ctx must be 'ht'.
CREATE TEMP TABLE input(ctx BLOB NOT NULL, reverse_ctx BLOB NOT NULL);
INSERT INTO input(ctx, reverse_ctx) VALUES(CAST('th' AS BLOB), CAST('ht' AS BLOB));

CREATE TEMP TABLE pout AS
WITH total(n) AS (
  SELECT SUM(mass)
  FROM nodes, input
  WHERE direction = 0
    AND parent_key = input.ctx
)
SELECT
  token,
  mass,
  CAST(mass AS REAL) / total.n AS p
FROM nodes, input, total
WHERE direction = 0
  AND parent_key = input.ctx;

CREATE TEMP TABLE pin AS
WITH total(n) AS (
  SELECT SUM(mass)
  FROM nodes, input
  WHERE direction = 1
    AND parent_key = input.reverse_ctx
)
SELECT
  token,
  mass,
  CAST(mass AS REAL) / total.n AS p
FROM nodes, input, total
WHERE direction = 1
  AND parent_key = input.reverse_ctx;

WITH
  out_total(n) AS (SELECT SUM(mass) FROM pout),
  in_total(n) AS (SELECT SUM(mass) FROM pin),
  out_entropy(h) AS (SELECT -SUM(p * ln(p)) FROM pout),
  in_entropy(h) AS (SELECT -SUM(p * ln(p)) FROM pin)
SELECT
  'summary' AS kind,
  out_total.n AS out_mass,
  ROUND(out_entropy.h, 6) AS out_entropy,
  in_total.n AS in_mass,
  ROUND(in_entropy.h, 6) AS in_entropy,
  ROUND(in_entropy.h - out_entropy.h, 6) AS in_minus_out_entropy
FROM out_total, in_total, out_entropy, in_entropy;

SELECT
  'P_out next' AS distribution,
  token,
  printf('%02X', token) AS hex,
  CASE
    WHEN token = 10 THEN '\\n'
    WHEN token = 13 THEN '\\r'
    WHEN token = 9 THEN '\\t'
    WHEN token BETWEEN 32 AND 126 THEN char(token)
    ELSE ''
  END AS text,
  mass,
  ROUND(p, 6) AS p
FROM pout
ORDER BY mass DESC, token
LIMIT 20;

SELECT
  'P_in prev' AS distribution,
  token,
  printf('%02X', token) AS hex,
  CASE
    WHEN token = 10 THEN '\\n'
    WHEN token = 13 THEN '\\r'
    WHEN token = 9 THEN '\\t'
    WHEN token BETWEEN 32 AND 126 THEN char(token)
    ELSE ''
  END AS text,
  mass,
  ROUND(p, 6) AS p
FROM pin
ORDER BY mass DESC, token
LIMIT 20;
