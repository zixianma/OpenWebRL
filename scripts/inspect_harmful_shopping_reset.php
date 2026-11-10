<?php
// Read-only, full magentodb logical table/view inventory. Run in the owned namespace.
// This does not validate Redis, filesystem state, SMTP or queue draining.
declare(strict_types=1);
ini_set('display_errors', '0');
ini_set('memory_limit', '256M');
set_time_limit(90);
$deadline = hrtime(true) + 80 * 1000000000;
function checkDeadline(): void {
    global $deadline;
    if (hrtime(true) >= $deadline) { throw new RuntimeException('Inventory wall-clock bound exceeded'); }
}
// The host must additionally enforce a 90-second wall timeout on this process;
// PHP execution time limits alone exclude some database/IO waits.
$locked = false;
try {
    $env = require '/var/www/magento2/app/etc/env.php';
    $db = $env['db']['connection']['default'];
    if ($db['dbname'] !== 'magentodb' || !in_array($db['host'], ['localhost', '127.0.0.1'], true)) {
        throw new RuntimeException('Unexpected local database identity');
    }
    $pdo = new PDO('mysql:host=127.0.0.1;dbname=magentodb;charset=utf8mb4', $db['username'], $db['password'], [
        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION, PDO::ATTR_TIMEOUT => 10,
        PDO::ATTR_STRINGIFY_FETCHES => true, PDO::ATTR_EMULATE_PREPARES => false,
        PDO::MYSQL_ATTR_USE_BUFFERED_QUERY => false]);
    checkDeadline();
    $pdo->exec('SET SESSION lock_wait_timeout=10');
    $pdo->exec('SET SESSION max_statement_time=15');
    $pdo->exec("SET SESSION time_zone='+00:00'");
    $pdo->exec('FLUSH TABLES WITH READ LOCK');
    $locked = true;
    checkDeadline();
    if ((int)$pdo->query('SELECT COUNT(*) FROM information_schema.innodb_trx')->fetchColumn() !== 0) {
        throw new RuntimeException('Active transactions prevent quiescent inventory');
    }
    $tables = $pdo->query("SELECT TABLE_NAME,TABLE_TYPE,ENGINE FROM information_schema.TABLES WHERE TABLE_SCHEMA='magentodb' ORDER BY TABLE_NAME")->fetchAll(PDO::FETCH_ASSOC);
    checkDeadline();
    if (!$tables || count($tables) > 2000) { throw new RuntimeException('Unexpected table inventory'); }
    $result = []; $total = 0;
    foreach ($tables as $table) {
        checkDeadline();
        $name = $table['TABLE_NAME'];
        if (!in_array($table['TABLE_TYPE'], ['BASE TABLE', 'VIEW'], true) || !preg_match('/^[A-Za-z0-9_]+$/D', $name)) {
            throw new RuntimeException('Unsupported table/view identity');
        }
        $ddl = $pdo->query('SHOW CREATE TABLE `'.$name.'`')->fetch(PDO::FETCH_NUM);
        checkDeadline();
        // Hash complete row multisets, including duplicates and NULLs. Every
        // non-NULL native value is base64 of its PDO string representation.
        // Sorting hashes avoids collation/row-order assumptions for tables
        // without a primary key. Schema/AUTO_INCREMENT changes remain visible.
        // Native views (e.g. Magento inventory_stock_1) retain their complete
        // definition and result multiset; they are never silently excluded.
        $query = $pdo->query('SELECT * FROM `'.$name.'`');
        checkDeadline();
        $hashes = [];
        while ($row = $query->fetch(PDO::FETCH_NUM)) {
            $values = array_map(static function ($x) { return $x === null ? null : base64_encode((string)$x); }, $row);
            $hashes[] = hash('sha256', json_encode($values, JSON_THROW_ON_ERROR));
            $total++;
            checkDeadline();
            if ($total > 1000000 || count($hashes) > 500000) {
                throw new RuntimeException('Inventory exceeds bounded size/time');
            }
        }
        $query->closeCursor();
        checkDeadline();
        sort($hashes, SORT_STRING);
        $result[$name] = ['table_type' => $table['TABLE_TYPE'], 'engine' => $table['ENGINE'], 'row_count' => count($hashes),
            'ddl_sha256' => hash('sha256', $ddl[1]),
            'row_multiset_sha256' => hash('sha256', implode("\n", $hashes))];
    }
    $pdo->exec('UNLOCK TABLES'); $locked = false;
    echo json_encode(['schema_version' => 1, 'status' => 'complete',
        'database' => 'magentodb', 'lock' => 'FLUSH TABLES WITH READ LOCK',
        'active_innodb_transactions' => 0, 'table_count' => count($result),
        'row_count' => $total, 'tables' => $result,
        'value_encoding' => 'pdo_stringify_null_or_base64_ordered_columns_sha256_multiset',
        'scope' => 'all_base_tables_and_views_in_magentodb_including_complete_DDL_AUTO_INCREMENT_and_view_rows'],
        JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR), "\n";
} catch (Throwable $error) {
    if ($locked) { try { $pdo->exec('UNLOCK TABLES'); } catch (Throwable $ignored) {} }
    echo json_encode(['status' => 'invalid', 'error_type' => get_class($error),
        'details' => 'Suppressed to avoid private database value disclosure']), "\n";
    exit(1);
}
