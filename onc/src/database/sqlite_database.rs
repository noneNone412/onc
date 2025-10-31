// using only sqlite3 as it is completely portable
// and is serverless so puts less load on the  cpu
/*
to do->
1) sqlite3 database must be  broken into several databases
   to avoid the massive database size,
2) sqlite3 database must avoid corruption on the process
   termination
 */

// b:: creating a safe sqlite3 database to avoid interrut corruption
//     use Write-Ahead Logging (WAL) isn't enabled.
/*
conn = sqlite3.connect("example.db")
conn.execute("PRAGMA journal_mode=WAL;")

import sqlite3
with sqlite3.connect("example.db") as conn:
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name) VALUES (?)", ("Alice",))
    # On interrupt or error, rollback is automatic

Signal-safe commit pattern (try/finally):
conn = sqlite3.connect("example.db")
try:
    conn.execute("BEGIN")
    conn.execute("INSERT INTO users (name) VALUES (?)", ("Alice",))
    conn.commit()
except Exception as e:
    conn.rollback()
    raise
finally:
    conn.close()


for rust
use WAL mode

use rusqlite::{Connection, OpenFlags};

let conn = Connection::open_with_flags(
    "my_db.sqlite",
    OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE
)?;
conn.pragma_update(None, "journal_mode", "WAL")?;

enable  synchronus writes
conn.pragma_update(None, "synchronous", "FULL")?; // Most durable
// or
conn.pragma_update(None, "synchronous", "NORMAL")?; // Good balance

transaction practices
// Always use transactions for writes
let tx = conn.transaction()?;
// perform your operations
tx.execute("INSERT INTO my_table (col) VALUES (?)", [value])?;
tx.commit()?; // Only at this point are changes actually written

*/
// e:: creating a safe sqlite3 database to avoid interrut corruption
//     use Write-Ahead Logging (WAL) isn't enabled.

/*
let path = env!("CARGO_MANIFEST_DIR");
let (before_last_slash, _) = path.rsplit_once('/').unwrap();
let full_path = format!("{}/data/{}.duckdb", before_last_slash, database_name);
println!("{}", full_path);
*/

use rusqlite::{Connection, OpenFlags};
pub struct SqliteDatabase {
    conn: rusqlite::Connection,
}

impl SqliteDatabase {
    pub fn new(databaseName: String) -> Self {
        let path = env!("CARGO_MANIFEST_DIR");
        let (before_last_slash, _) = path.rsplit_once('/').unwrap();
        let full_path = format!("{}/data/{}.db", before_last_slash, databaseName);
        println!("{}", &full_path);
        let conn_: Result<Connection, rusqlite::Error> = Connection::open_with_flags(
            &full_path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        );
        if !conn_.is_ok() {
            panic!("Failed to connect to database in {}", full_path)
        }
        conn_
            .as_ref()
            .unwrap()
            .pragma_update(None, "journal_mode", "WAL");
        conn_
            .as_ref()
            .unwrap()
            .pragma_update(None, "journal_mode", "WAL");
        SqliteDatabase {
            conn: conn_.unwrap(),
        }
    }
    pub fn connection(&self) -> &rusqlite::Connection {
        &self.conn
    }
}
