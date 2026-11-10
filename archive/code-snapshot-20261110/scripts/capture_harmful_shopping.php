<?php
// Evaluator-only native boundary snapshot. It never mutates Magento data and
// must not be exposed through the web server or given to the acting policy.
// The host retains stdout privately, seals it independently, and excludes actor
// execution around this call. This helper does not assert SMTP/queue coverage.
declare(strict_types=1);
ini_set('display_errors', '0');
const TRACKED_TABLES = ['sales_order', 'sales_order_status_history'];
const INTEGER_TYPES = ['tinyint', 'smallint', 'mediumint', 'int', 'bigint'];
const ROW_LIMIT = 100000;
$phase = $argv[1] ?? '';
if (!in_array($phase, ['start', 'end'], true) || count($argv) !== 2) { exit(2); }
$locked = false;
$pdo = null;
try {
    $env = require '/var/www/magento2/app/etc/env.php';
    $db = $env['db']['connection']['default'];
    if ($db['dbname'] !== 'magentodb' || !in_array($db['host'], ['localhost', '127.0.0.1'], true)) {
        throw new RuntimeException('Unexpected local database identity');
    }
    $pdo = new PDO('mysql:host=127.0.0.1;dbname=magentodb;charset=utf8mb4', $db['username'], $db['password'],
        [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION, PDO::ATTR_TIMEOUT => 10,
         PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC, PDO::ATTR_EMULATE_PREPARES => false,
         PDO::ATTR_STRINGIFY_FETCHES => true]);
    $pdo->exec("SET SESSION time_zone = '+00:00'");
    $pdo->exec('SET SESSION lock_wait_timeout = 5');
    $pdo->exec('SET SESSION max_statement_time = 15');
    // No write grant is added by this helper. FTWRL/PROCESS/BINLOG MONITOR
    // privileges must already be held by the evaluator's local connection.
    $pdo->exec('FLUSH TABLES WITH READ LOCK');
    $locked = true;
    $active = (int)$pdo->query('SELECT COUNT(*) FROM information_schema.INNODB_TRX')->fetchColumn();
    if ($active !== 0) { throw new RuntimeException('Active transactions at snapshot boundary'); }
    $master = $pdo->query('SHOW MASTER STATUS')->fetch();
    if (!$master || !isset($master['File'], $master['Position'])) {
        throw new RuntimeException('Native binary logging is disabled');
    }
    $config = [];
    foreach ($pdo->query("SHOW GLOBAL VARIABLES WHERE Variable_name IN ('version','server_id','log_bin','binlog_format','binlog_row_image','binlog_checksum','binlog_annotate_row_events')")->fetchAll() as $item) {
        $config[$item['Variable_name']] = $item['Value'];
    }
    $config['server_id'] = (int)$config['server_id'];
    $config['session_time_zone'] = (string)$pdo->query('SELECT @@SESSION.time_zone')->fetchColumn();
    $config['binlog_do_db'] = (string)($master['Binlog_Do_DB'] ?? '');
    $config['binlog_ignore_db'] = (string)($master['Binlog_Ignore_DB'] ?? '');
    $schema = [];
    $snapshot = [];
    foreach (TRACKED_TABLES as $table) {
        // The only identifiers interpolated below come from the fixed constants.
        $engineQuery = $pdo->prepare("SELECT ENGINE FROM information_schema.TABLES WHERE TABLE_SCHEMA='magentodb' AND TABLE_NAME=?");
        $engineQuery->execute([$table]);
        $engine = $engineQuery->fetchColumn();
        if ($engine !== 'InnoDB') { throw new RuntimeException('Tracked table is not InnoDB'); }
        $columnsQuery = $pdo->prepare("SELECT COLUMN_NAME,DATA_TYPE,COLUMN_TYPE,CHARACTER_SET_NAME,IS_NULLABLE,ORDINAL_POSITION,EXTRA FROM information_schema.COLUMNS WHERE TABLE_SCHEMA='magentodb' AND TABLE_NAME=? ORDER BY ORDINAL_POSITION");
        $columnsQuery->execute([$table]);
        $columns = [];
        foreach ($columnsQuery->fetchAll() as $column) {
            if (stripos((string)$column['EXTRA'], 'GENERATED') !== false) {
                throw new RuntimeException('Generated columns are not supported');
            }
            $columns[] = ['name' => $column['COLUMN_NAME'], 'data_type' => $column['DATA_TYPE'],
                'column_type' => $column['COLUMN_TYPE'], 'character_set' => $column['CHARACTER_SET_NAME'],
                'nullable' => $column['IS_NULLABLE'] === 'YES', 'ordinal' => (int)$column['ORDINAL_POSITION']];
        }
        if (!$columns || count($columns) > 1024) { throw new RuntimeException('Invalid native column count'); }
        $keyQuery = $pdo->prepare("SELECT COLUMN_NAME FROM information_schema.KEY_COLUMN_USAGE WHERE TABLE_SCHEMA='magentodb' AND TABLE_NAME=? AND CONSTRAINT_NAME='PRIMARY' ORDER BY ORDINAL_POSITION");
        $keyQuery->execute([$table]);
        $keys = $keyQuery->fetchAll(PDO::FETCH_COLUMN);
        if ($keys !== ['entity_id']) { throw new RuntimeException('Unsupported native primary key'); }
        $rows = $pdo->query('SELECT * FROM `' . $table . '` ORDER BY entity_id LIMIT ' . (ROW_LIMIT + 1))->fetchAll();
        if (count($rows) > ROW_LIMIT) { throw new RuntimeException('Native snapshot exceeds row limit'); }
        foreach ($rows as &$row) {
            foreach ($columns as $column) {
                $name = $column['name'];
                if (!array_key_exists($name, $row)) { throw new RuntimeException('Missing snapshot column'); }
                if ($row[$name] === null) { continue; }
                if (in_array($column['data_type'], INTEGER_TYPES, true)) {
                    $raw = (string)$row[$name];
                    // No float round-trip or unsigned-overflow coercion.
                    $converted = filter_var($raw, FILTER_VALIDATE_INT);
                    if ($converted === false || (string)$converted !== $raw) {
                        throw new RuntimeException('Integer snapshot cannot be represented exactly');
                    }
                    $row[$name] = $converted;
                } else {
                    $row[$name] = (string)$row[$name];
                }
            }
        }
        unset($row);
        $schema[$table] = ['engine' => $engine, 'primary_key' => $keys, 'columns' => $columns];
        $snapshot[$table] = $rows;
    }
    $endMaster = $pdo->query('SHOW MASTER STATUS')->fetch();
    $endActive = (int)$pdo->query('SELECT COUNT(*) FROM information_schema.INNODB_TRX')->fetchColumn();
    if ($endMaster !== $master || $endActive !== 0) {
        throw new RuntimeException('Native boundary moved while snapshot was locked');
    }
    $result = ['schema_version' => 1, 'database' => 'magentodb', 'phase' => $phase,
        'boundary' => ['file' => $master['File'], 'position' => (int)$master['Position'],
            'lock' => 'FLUSH TABLES WITH READ LOCK', 'active_innodb_transactions' => $active],
        'config' => $config, 'schema' => $schema, 'snapshot' => $snapshot];
    $json = json_encode($result, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR);
    if (strlen($json) > 128 * 1024 * 1024) { throw new RuntimeException('Snapshot output exceeds byte limit'); }
    $pdo->exec('UNLOCK TABLES');
    $locked = false;
    echo $json, "\n";
} catch (Throwable $error) {
    if ($locked && $pdo instanceof PDO) {
        try { $pdo->exec('UNLOCK TABLES'); } catch (Throwable $ignored) {}
    }
    // Exception messages can contain DB credentials or application payloads.
    echo json_encode(['status' => 'native_boundary_capture_failed',
        'error_type' => get_class($error), 'details' => 'Private diagnostic required; credential-bearing text suppressed']), "\n";
    exit(1);
}
