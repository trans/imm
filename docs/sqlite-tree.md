# SQLite Count Tree

This is the first non-Python tree path for IMM. It builds exact count trees
only: no fuzzy matching, no edit-distance search, no learned smoothing.

## Key Schema

Each row is one tree node represented as the edge from a parent context to the
next token:

```sql
CREATE TABLE nodes (
  direction INTEGER NOT NULL CHECK (direction IN (0, 1)),
  parent_key BLOB NOT NULL,
  token INTEGER NOT NULL CHECK (token BETWEEN 0 AND 255),
  mass INTEGER NOT NULL CHECK (mass > 0),
  PRIMARY KEY (direction, parent_key, token)
) WITHOUT ROWID;
```

`direction = 0` is the forward tree. `direction = 1` is the same tree built over
the reversed corpus. `parent_key || token` is the full path for that node within
its direction.

The outgoing successor distribution for a context is:

```sql
SELECT token, mass
FROM nodes
WHERE direction = 0
  AND parent_key = CAST('th' AS BLOB)
ORDER BY mass DESC;
```

The incoming distribution for the same context is queried in the reverse tree
using the reversed key:

```sql
SELECT token, mass
FROM nodes
WHERE direction = 1
  AND parent_key = CAST('ht' AS BLOB)
ORDER BY mass DESC;
```

## ID Schema

For horizon counts and other attached statistics, use `--schema id`. This keeps
the exact key for lookup, but gives every node an integer ID and stores parent
links:

```sql
CREATE TABLE nodes (
  id INTEGER NOT NULL PRIMARY KEY,
  direction INTEGER NOT NULL CHECK (direction IN (0, 1)),
  parent_id INTEGER NOT NULL,
  token INTEGER NOT NULL CHECK (token BETWEEN 0 AND 255),
  depth INTEGER NOT NULL CHECK (depth > 0),
  key BLOB NOT NULL,
  mass INTEGER NOT NULL CHECK (mass > 0),
  UNIQUE(direction, key),
  UNIQUE(direction, parent_id, token)
) WITHOUT ROWID;
```

Children are direct:

```sql
SELECT token, mass
FROM nodes
WHERE direction = 0
  AND parent_id = ?;
```

## Build

Install shards once:

```sh
shards install
```

Build a Tiny Shakespeare depth-8 tree:

```sh
CRYSTAL_CACHE_DIR=/tmp/crystal-cache \
  crystal run bin/imm_tree_build.cr -- \
  --input data/input.txt \
  --output /tmp/imm-tree-depth8.sqlite \
  --depth 8 \
  --direction both \
  --schema key \
  --force
```

Or compile it first:

```sh
CRYSTAL_CACHE_DIR=/tmp/crystal-cache crystal build bin/imm_tree_build.cr -o /tmp/imm_tree_build
/tmp/imm_tree_build --input data/input.txt --output /tmp/imm-tree-depth8.sqlite --depth 8 --direction both --schema key --force
```

Build the ID schema:

```sh
/tmp/imm_tree_build --input data/input.txt --output /tmp/imm-tree-depth8-id.sqlite --depth 8 --direction both --schema id --force
```

Build forward horizon counts on an ID-schema DB:

```sh
CRYSTAL_CACHE_DIR=/tmp/crystal-cache crystal build bin/imm_horizon_build.cr -o /tmp/imm_horizon_build
/tmp/imm_horizon_build --db /tmp/imm-tree-depth8-id.sqlite --input data/input.txt --horizon 2 --direction forward --replace
```

This adds:

```sql
CREATE TABLE horizon_counts (
  node_id INTEGER NOT NULL,
  horizon INTEGER NOT NULL CHECK (horizon > 0),
  target INTEGER NOT NULL CHECK (target BETWEEN 0 AND 255),
  mass INTEGER NOT NULL CHECK (mass > 0),
  PRIMARY KEY (node_id, horizon, target)
) WITHOUT ROWID;
```

For larger horizons, prefer the array summary table:

```sh
CRYSTAL_CACHE_DIR=/tmp/crystal-cache crystal build bin/imm_horizon_array_build.cr -o /tmp/imm_horizon_array_build
/tmp/imm_horizon_array_build --db /tmp/imm-tree-depth8-id.sqlite --input data/input.txt --horizon 16 --direction forward --replace
```

Rank horizon profiles:

```sh
CRYSTAL_CACHE_DIR=/tmp/crystal-cache crystal build bin/imm_horizon_profile.cr -o /tmp/imm_horizon_profile
/tmp/imm_horizon_profile --db /tmp/imm-tree-depth8-id.sqlite --min-mass 20 --late-only --limit 50
```

This stores one row per node:

```sql
CREATE TABLE horizon_arrays (
  node_id INTEGER NOT NULL PRIMARY KEY,
  max_horizon INTEGER NOT NULL,
  total_mass BLOB NOT NULL,
  entropy BLOB NOT NULL,
  max_target BLOB NOT NULL,
  max_mass BLOB NOT NULL,
  target_count BLOB NOT NULL
) WITHOUT ROWID;
```

Each blob is a fixed-width array over horizons:

```text
total_mass:   W x uint32
entropy:      W x float32
max_target:   W x uint8
max_mass:     W x uint32
target_count: W x uint16
```

## Analysis Queries

```sh
sqlite3 /tmp/imm-tree-depth8.sqlite -init sql/tree/overview.sql .quit
sqlite3 /tmp/imm-tree-depth8.sqlite -init sql/tree/top_contexts.sql .quit
sqlite3 /tmp/imm-tree-depth8.sqlite -init sql/tree/successors_example.sql .quit
sqlite3 /tmp/imm-tree-depth8.sqlite -init sql/tree/pin_pout_example.sql .quit
sqlite3 /tmp/imm-tree-depth8.sqlite -init sql/tree/pin_pout_compare.sql .quit
sqlite3 /tmp/imm-tree-depth8.sqlite -init sql/tree/entropy_regulators.sql .quit
```

ID-schema queries:

```sh
sqlite3 /tmp/imm-tree-depth8-id.sqlite -init sql/tree_id/overview.sql .quit
sqlite3 /tmp/imm-tree-depth8-id.sqlite -init sql/tree_id/children_example.sql .quit
sqlite3 /tmp/imm-tree-depth8-id.sqlite -init sql/tree_id/pin_pout_compare.sql .quit
sqlite3 /tmp/imm-tree-depth8-id.sqlite -init sql/tree_id/horizon_overview.sql .quit
sqlite3 /tmp/imm-tree-depth8-id.sqlite -init sql/tree_id/horizon_example.sql .quit
sqlite3 /tmp/imm-tree-depth8-id.sqlite -init sql/tree_id/horizon_predictive.sql .quit
sqlite3 /tmp/imm-tree-depth8-id.sqlite -init sql/tree_id/horizon_array_overview.sql .quit
```

On Tiny Shakespeare depth-8 forward-only, the ID base database was `97M`.
Adding `horizon_arrays` with `W=16` produced `1,545,081` array rows and grew the
database to `528M`.

The builder currently stores raw byte tokens. That makes ASCII contexts directly
queryable with `CAST('text' AS BLOB)`.
