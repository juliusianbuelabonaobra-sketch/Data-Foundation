import pyodbc
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
import logging
import sys
import time
import datetime
import os
import glob
from decimal import Decimal

# Load YAML config
with open("/DataFoundation/MMS/SMR/Raw/SMR_config.yaml", "r") as file:
    config = yaml.safe_load(file)

today_str = datetime.datetime.today().strftime("%Y%m%d")

# Use fixed Exports and Logs folders
exports_dir = config.get("exports_dir", "Exports")
logs_dir = config.get("logs_dir", "Logs")

os.makedirs(exports_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Add date-stamp to log file
log_base = os.path.splitext(config.get("log_file", "export.log"))[0]
log_file = os.path.join(logs_dir, f"{log_base}_{today_str}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
)

# Build connection string
conn_cfg = config["conn"]
conn_str = (
    f"DRIVER={{{conn_cfg['driver']}}};"
    f"SYSTEM={conn_cfg['system']};"
    f"PORT={conn_cfg['port']};"
    f"UID={conn_cfg['uid']};"
    f"PWD={conn_cfg['pwd']};"
    f"DATABASE={conn_cfg['database']};"
)

def load_sql_files(sql_directory, file_pattern="*.yaml"):
    """Load all YAML query files from directory."""
    pattern = os.path.join(sql_directory, file_pattern)
    return sorted(glob.glob(pattern))

def get_output_filename(base, fmt, timestamp):
    """Generate output filename with optional timestamp."""
    if timestamp:
        ts = datetime.datetime.now().strftime("%Y%m%d")
        name = f"{base}_{ts}.{fmt}"
    else:
        name = f"{base}.{fmt}"
    return os.path.join(exports_dir, name)

def normalize_decimal_schema(table):
    """
    Normalize all decimal fields to consistent precision for Qlik compatibility.
    This solves the IBM i precision inconsistency issue.
    """
    new_arrays = []
    new_fields = []
    
    for i, field in enumerate(table.schema):
        column_array = table.column(i)
        field_type = str(field.type)
        
        if field_type.startswith('decimal128'):
            # Convert all decimals to string for consistent handling
            # Qlik can handle string-to-number conversion better than inconsistent decimals
            try:
                # Convert to string to avoid precision issues
                string_array = pa.compute.cast(column_array, pa.string())
                new_arrays.append(string_array)
                new_fields.append(pa.field(field.name, pa.string()))
                
            except Exception as e:
                logging.warning(f"Failed to convert decimal column {field.name}: {e}")
                # Fallback: keep as-is
                new_arrays.append(column_array)
                new_fields.append(field)
        else:
            # Keep non-decimal fields as-is
            new_arrays.append(column_array)
            new_fields.append(field)
    
    return pa.table(new_arrays, schema=pa.schema(new_fields))

def process_query_optimized(cursor, query, chunk_size=100000):
    """
    Optimized chunked processing with consistent schema handling for IBM i.
    """
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    
    chunk_count = 0
    total_rows = 0
    master_schema = None
    
    while True:
        rows = cursor.fetchmany(chunk_size)
        if not rows:
            break
            
        chunk_count += 1
        total_rows += len(rows)
        
        # Progress logging every 5 chunks
        if chunk_count % 5 == 0:
            logging.info(f"Processing chunk {chunk_count}: {total_rows:,} rows processed")
        
        if rows:
            # Create PyArrow table from chunk
            data = list(zip(*rows))
            table = pa.table([pa.array(col) for col in data], names=columns)
            
            # Normalize decimal schema for consistency
            normalized_table = normalize_decimal_schema(table)
            
            if master_schema is None:
                # Establish master schema from first chunk
                master_schema = normalized_table.schema
                logging.info(f"Schema established with {len(master_schema)} columns")
                
            # Ensure chunk matches master schema
            if normalized_table.schema != master_schema:
                try:
                    # Try to cast to master schema
                    normalized_table = normalized_table.cast(master_schema)
                except Exception as e:
                    logging.warning(f"Schema cast failed for chunk {chunk_count}: {e}")
                    # Continue with current schema - let parquet handle it
            
            yield normalized_table
    
    logging.info(f"Processing complete: {total_rows:,} rows in {chunk_count} chunks")

def write_optimized_parquet(cursor, query, output_path, compression="snappy", max_file_size_mb=None, use_timestamp=False, base_name="Data"):
    """
    Write parquet files optimized for Qlik Sense consumption.
    """
    written_files = []
    
    if not max_file_size_mb:
        # Single file approach (recommended for Qlik)
        ts = datetime.datetime.now().strftime("%Y%m%d") if use_timestamp else ""
        final_name = f"{base_name}_{ts}.parquet" if ts else f"{base_name}.parquet"
        final_path = os.path.join(exports_dir, final_name)
        
        writer = None
        try:
            for chunk_table in process_query_optimized(cursor, query):
                if writer is None:
                    # Initialize writer with first chunk schema
                    writer = pq.ParquetWriter(final_path, chunk_table.schema, compression=compression)
                    logging.info(f"Started writing to: {final_path}")
                
                writer.write_table(chunk_table)
            
            if writer:
                writer.close()
                written_files.append(final_path)
                file_size_mb = os.path.getsize(final_path) / (1024 * 1024)
                logging.info(f"? Completed: {final_path} ({file_size_mb:.1f} MB)")
                
        except Exception as e:
            if writer:
                writer.close()
            raise e
            
    else:
        # Multi-file approach for very large datasets
        max_bytes = int(max_file_size_mb) * 1024 * 1024
        current_file_size = 0
        part = 1
        writer = None
        ts = datetime.datetime.now().strftime("%Y%m%d") if use_timestamp else ""
        master_schema = None
        
        try:
            for chunk_table in process_query_optimized(cursor, query):
                chunk_size = chunk_table.nbytes
                
                # Start new file if needed
                if writer is None or (current_file_size + chunk_size) > max_bytes:
                    if writer:
                        writer.close()
                        prev_size_mb = os.path.getsize(written_files[-1]) / (1024 * 1024)
                        logging.info(f"? Part {part-1} complete: {prev_size_mb:.1f} MB")
                    
                    fname = f"{base_name}_part{part}_{ts}.parquet" if ts else f"{base_name}_part{part}.parquet"
                    part_path = os.path.join(exports_dir, fname)
                    
                    # Use established schema or current chunk schema
                    schema_to_use = master_schema if master_schema else chunk_table.schema
                    if master_schema is None:
                        master_schema = chunk_table.schema
                    
                    writer = pq.ParquetWriter(part_path, schema_to_use, compression=compression)
                    written_files.append(part_path)
                    current_file_size = 0
                    part += 1
                    logging.info(f"Started part {part-1}: {part_path}")
                
                writer.write_table(chunk_table)
                current_file_size += chunk_size
            
            if writer:
                writer.close()
                if written_files:
                    final_size_mb = os.path.getsize(written_files[-1]) / (1024 * 1024)
                    logging.info(f"? Final part complete: {final_size_mb:.1f} MB")
                
        except Exception as e:
            if writer:
                writer.close()
            raise e
    
    return written_files

# ===== MAIN EXECUTION =====
def main():
    try:
        logging.info("=== IBM i to Parquet Extraction Started ===")
        logging.info("Connecting to database...")
        conn = pyodbc.connect(conn_str)
        logging.info("? Database connection established")

        queries_dir = config["queries_dir"]
        query_files = load_sql_files(queries_dir, "*.yaml")

        if not query_files:
            logging.error("? No query files found")
            sys.exit(1)

        logging.info(f"Found {len(query_files)} query file(s):")
        for qf in query_files:
            logging.info(f"  - {os.path.basename(qf)}")

        # Process each query file
        for qf in query_files:
            with open(qf, "r") as f:
                qcfg = yaml.safe_load(f)

            try:
                name = qcfg["name"]
                query = qcfg["query"]
                fmt = qcfg.get("format", "parquet").lower()
                base_output = qcfg["output"]
                compression = qcfg.get("compression", "snappy")
                use_timestamp = qcfg.get("timestamp", False)
                max_file_size_mb = qcfg.get("max_file_size_mb")

                # Only support parquet for now
                if fmt != "parquet":
                    logging.warning(f"?? Skipping {name}: Only parquet format supported")
                    continue

                # Replace database placeholder
                query = query.replace("${database}", conn_cfg["database"])

                logging.info(f"\n--- Processing: {name} ---")
                logging.info(f"Format: {fmt.upper()}")
                logging.info(f"Compression: {compression}")
                if max_file_size_mb:
                    logging.info(f"Max file size: {max_file_size_mb}MB")
                
                start_time = time.time()
                cursor = conn.cursor()
                
                # Get output path
                output_path = get_output_filename(base_output, fmt, use_timestamp)

                # Process query
                output_files = write_optimized_parquet(
                    cursor, query, output_path, compression, 
                    max_file_size_mb, use_timestamp, base_output
                )

                duration = round(time.time() - start_time, 2)
                total_size_mb = sum(os.path.getsize(f) / (1024 * 1024) for f in output_files)
                
                logging.info(f"? '{name}' completed!")
                logging.info(f"   Duration: {duration}s")
                logging.info(f"   Files: {len(output_files)}")
                logging.info(f"   Total size: {total_size_mb:.1f} MB")
                
                cursor.close()

            except Exception as query_error:
                logging.error(f"? Error processing '{qcfg.get('name', 'unknown')}': {query_error}")
                if 'cursor' in locals():
                    cursor.close()

        logging.info("\n=== ? Extraction Process Completed ===")

    except pyodbc.Error as e:
        logging.critical(f"? Database connection failed: {e}")
        sys.exit(1)

    except Exception as e:
        logging.critical(f"? Unexpected error: {e}")
        sys.exit(1)

    finally:
        if 'conn' in locals():
            conn.close()
            logging.info("Database connection closed")

if __name__ == "__main__":
    main()