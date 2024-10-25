import yaml
import pymysql 
import argparse
import sys
import json
import yaml

def get_access_token():
    from kubernetes import client, config

    # Load in-cluster config (uses service account's token to authenticate)
    # config.load_incluster_config()  # Use this inside the cluster
    # Load the kubeconfig (uses ~/.kube/config or any provided kubeconfig file)
    config.load_kube_config()  # Uncomment if running outside the cluster
    
    # Create an API client
    v1 = client.CoreV1Api()
    
    # Fetch the secret by name and namespace
    namespace = "scitix-system"
    secret_name = "hisys-mysql-secret"
    
    try:
        secret = v1.read_namespaced_secret(secret_name, namespace)
        username = secret.data['username']  # Base64 encoded
        password = secret.data['password']  # Base64 encoded
        host = secret.data['host']  # Base64 encoded
        port = secret.data['port']  # Base64 encoded
    
        # Decode the base64 encoded values
        import base64
        decoded_username = base64.b64decode(username).decode('utf-8')
        decoded_password = base64.b64decode(password).decode('utf-8')
        decoded_host = base64.b64decode(host).decode('utf-8')
        decoded_port = base64.b64decode(port).decode('utf-8')
    
        print(f"Username: {decoded_username}")
        print(f"Password: {decoded_password}")
        print(f"Host: {decoded_host}")
        print(f"Port: {decoded_port}")
    
    except client.exceptions.ApiException as e:
        print(f"Exception when reading secret: {e}")
    return decoded_username, decoded_password, decoded_host, decoded_port


# Function to connect to MySQL
def connect_to_mysql(cluster=None):
    try:
        user, passwd, host, port = get_access_token()
        # Connect to MySQL (replace with your actual connection details)
        connection = pymysql.connect(
            host=host,
            port=int(port),
            user=user,
            password=passwd,
            database='scitix_hisys',
        )
        print("Connected to MySQL scitix_hisys database successfully!")
        return connection
    except Exception as e:
        print(f"Error connecting to MySQL scitix_hisys : {e}")
        sys.exit(-1)

# Function to create or update cluster and machine tables
def create_or_update_cluster_tables_from_yaml(spec_yaml):
    connection = connect_to_mysql()
    try:
        with open(spec_yaml, 'r') as file:
            spec_datas = yaml.safe_load(file)
            
        with connection.cursor() as cursor:
            # Create cluster table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cluster (
                    name VARCHAR(255) PRIMARY KEY,
                    description TEXT,
                    UNIQUE (name)
                )
            """)
            print("Created or updated cluster table")

            # Create cluster_services_metrics table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cluster_services_metrics (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    cluster_name VARCHAR(255),
                    service_name VARCHAR(50) NOT NULL,
                    metric_name VARCHAR(255) NOT NULL,
                    metric_value VARCHAR(255),
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX (cluster_name, service_name, metric_name),
                    FOREIGN KEY (cluster_name) REFERENCES cluster(name) ON DELETE CASCADE,
                    UNIQUE (cluster_name, service_name, metric_name)
                )
            """)
            print("Created or updated metrics table")

            # Insert or update data from YAML
            for spec_type, spec_data in spec_datas.items():
                if spec_type == 'service_metric_spec':
                    for cluster_spec_data in spec_data:
                        cluster_name = cluster_spec_data['cluster_name']
                        machine_description = cluster_spec_data['description']

                        # Insert or update cluster_services_metrics table
                        cursor.execute("""
                            INSERT INTO cluster (name, description)
                            VALUES (%s, %s)
                            ON DUPLICATE KEY UPDATE description = VALUES(description)
                        """, (cluster_name, machine_description))
                        print(f'Inserted or updated cluster_services_metrics table: {cluster_name}')
                        
                        # if cursor.lastrowid:
                        #     system_id = cursor.lastrowid
                        # else:
                        #     cursor.execute("SELECT id FROM cluster_services_metrics WHERE name = %s", (cluster_name))
                        #     system_id = cursor.fetchone()[0]
                        # print(f'Start to insert or update metrics')
                        metrics = cluster_spec_data['metrics']
                        for service_name, sub_metircs in metrics.items():
                            for metric_name, metric_value in sub_metircs.items():
                                # Insert or update metrics
                                # print(f'Start to insert or update metrics for "{metric_name}": {metric_value}')
                                cursor.execute("""
                                    INSERT INTO cluster_services_metrics (cluster_name, service_name, metric_name, metric_value)
                                    VALUES (%s, %s, %s, %s)
                                    ON DUPLICATE KEY UPDATE metric_value = VALUES(metric_value)
                                """, (cluster_name, service_name, metric_name, metric_value))
                                print(f'Insert or update metrics completed for "{metric_name}": {metric_value}')

            connection.commit()
            print("Data insertion or update completed!")

    except MySQLdb.Error as e:
        print(f"Error creating or updating tables: {e}")

    connection.close()
    print("Connection to MySQL closed")        
        
# Function to create or update cluster and machine tables
def create_or_update_machine_tables_from_yaml(spec_yaml):
    connection = connect_to_mysql()

    try:
        with open(spec_yaml, 'r') as file:
            spec_datas = yaml.safe_load(file)
            
        with connection.cursor() as cursor:
            # Create machine table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS machine (
                    name VARCHAR(255) PRIMARY KEY,
                    description TEXT,
                    UNIQUE (name)
                )
            """)
            print("Created or updated machine table")

            # Create machine_metrics table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS machine_metrics (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    machine_name VARCHAR(255),
                    metric_type VARCHAR(50) NOT NULL,
                    metric_name VARCHAR(255) NOT NULL,
                    metric_value VARCHAR(255),
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX (machine_name, metric_type, metric_name),
                    FOREIGN KEY (machine_name) REFERENCES machine(name) ON DELETE CASCADE,
                    UNIQUE (machine_name, metric_type, metric_name)
                )
            """)
            print("Created or updated metrics table")

            # Insert or update data from YAML
            for spec_type, spec_data in spec_datas.items():
                if spec_type == 'compute_metric_spec':
                    for machine_spec_data in spec_data:
                        machine_name = machine_spec_data['machine_name']
                        machine_description = machine_spec_data['description']

                        # Insert or update machine table
                        cursor.execute("""
                            INSERT INTO machine (name, description)
                            VALUES (%s, %s)
                            ON DUPLICATE KEY UPDATE description = VALUES(description)
                        """, (machine_name, machine_description))
                        print(f'Inserted or updated machine table: {machine_name}')
                        
                        # if cursor.lastrowid:
                        #     system_id = cursor.lastrowid
                        # else:
                        #     cursor.execute("SELECT id FROM machine WHERE name = %s", (machine_name))
                        #     system_id = cursor.fetchone()[0]
                        # print(f'Start to insert or update metrics')
                        metrics = machine_spec_data['metrics']
                        for metric_type, sub_metircs in metrics.items():
                            for metric_name, metric_value in sub_metircs.items():
                                # Insert or update metrics
                                # print(f'Start to insert or update metrics for "{metric_name}": {metric_value}')
                                cursor.execute("""
                                    INSERT INTO machine_metrics (machine_name, metric_type, metric_name, metric_value)
                                    VALUES (%s, %s, %s, %s)
                                    ON DUPLICATE KEY UPDATE metric_value = VALUES(metric_value)
                                """, (machine_name, metric_type, metric_name, metric_value))
                                print(f'Insert or update metrics completed for "{metric_name}": {metric_value}')

            connection.commit()
            print("Data insertion or update completed!")

    except MySQLdb.Error as e:
        print(f"Error creating or updating tables: {e}")

    connection.close()
    print("Connection to MySQL closed")        

def query_cluster_metrics(cluster_name, output_format='json', output_file=None):
    connection = connect_to_mysql()
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT service_name, metric_name, metric_value
            FROM cluster_services_metrics
            WHERE cluster_name = %s
        """, (cluster_name,))
        metrics = cursor.fetchall()
    connection.close()
    print(f'Querying metrics completed for "{cluster_name}": {metrics}')
    return sql_metrics_output_to_yaml(cluster_name, metrics, output_format, output_file)

def query_machine_metrics(machine_name, output_format='json', output_file=None):
    connection = connect_to_mysql()
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT metric_type, metric_name, metric_value
            FROM machine_metrics
            WHERE machine_name = %s
        """, (machine_name,))
        metrics = cursor.fetchall()
    connection.close()
    print(f'Querying metrics completed for "{machine_name}": {metrics}')
    return sql_metrics_output_to_yaml(machine_name, metrics, output_format, output_file)


def sql_metrics_output_to_yaml(key_name, metrics, output_format='yaml', output_file=None):
    print(f'Querying metrics completed for "{key_name}": {metrics}')
    metrics_dict = {}
    for row in metrics:
        # print(f'row: {row}')
        metric_type, metric_name, metric_value = row
        if metric_type not in metrics_dict:
            metrics_dict[metric_type] = {}
        metrics_dict[metric_type][metric_name] = metric_value
    output_dict = metrics_dict #{key_name: metrics_dict}
    # print(f'{output_dict=}')
    if output_file is None:
        return output_dict
    else:
        with open(output_file, 'w') as file:
            if output_format == 'json':
                json.dump(output_dict, file)
            elif output_format == 'yaml':    
                yaml.dump(output_dict, file, default_flow_style=False)
        return 0

# Function to create or update cluster and machine tables
def create_or_update_testset_tables_from_yaml(spec_yaml):
    connection = connect_to_mysql()

    try:
        with open(spec_yaml, 'r') as file:
            testset_datas = yaml.safe_load(file)
            
        with connection.cursor() as cursor:
            # Create hisys_testset table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hisys_testset (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    testset VARCHAR(255),
                    test_type VARCHAR(50) NOT NULL,
                    test_name VARCHAR(255) NOT NULL,
                    test_cmd VARCHAR(255),
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY (testset, test_type, test_name)
                )
            """)
            print("Created or updated hisys_testset table")

            # Insert or update data from YAML
            for testset_type, testset_type_sets in testset_datas.items():
                for testset_class, testsets in testset_type_sets.items():
                    for testset in testsets:
                        test_name = testset['test_name']
                        test_cmd = testset['test_cmd']
                        # Insert or update hisys_testset
                        # print(f'Start to insert or update metrics for {testset_type} : {testset_class} : {test_name}: {test_cmd}')
                        cursor.execute("""
                            INSERT INTO hisys_testset (testset, test_type, test_name, test_cmd)
                            VALUES (%s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE test_cmd = VALUES(test_cmd)
                        """, (testset_type, testset_class, test_name, test_cmd))
                        print(f'Insert or update hisys_testset table completed for "{test_name}": {test_cmd}')

            connection.commit()
            print("Data insertion or update completed!")

    except MySQLdb.Error as e:
        print(f"Error creating or updating tables: {e}")

    connection.close()
    print("Connection to MySQL closed")        

def query_testset(testset, test_type=None, test_name=None, output_format='json', output_file=None):
    connection = connect_to_mysql()
    with connection.cursor() as cursor:
        if testset is not None and test_type is not None and test_name is not None:
            print(f'testset={testset}, test_type={test_type}, test_name={test_name}')
            cursor.execute("""
                SELECT test_type, test_name, test_cmd
                FROM hisys_testset
                WHERE testset = %s AND test_type = %s AND test_name = %s
            """, (testset, test_type, test_name,))
            testsets = cursor.fetchall()
        elif testset is not None and test_type is not None:
            print(f'testset={testset}, test_type={test_type}')
            cursor.execute("""
                SELECT test_type, test_name, test_cmd
                FROM hisys_testset
                WHERE testset = %s AND test_type = %s
            """, (testset, test_type,))
            testsets = cursor.fetchall()
        elif testset is not None:
            print(f'testset={testset}')
            cursor.execute("""
                SELECT test_type, test_name, test_cmd
                FROM hisys_testset
                WHERE testset = %s
            """, (testset,))
            testsets = cursor.fetchall()
        else:
            cursor.execute("""
                SELECT test_type, test_name, test_cmd
                FROM hisys_testset
            """)
            testsets = cursor.fetchall()
    connection.close()
    print(testsets)
    print(f'Querying testset completed for {testset=} : {test_type=} : {test_name=}')
    return sql_testset_output_to_yaml(testsets, output_format, output_file)

def sql_testset_output_to_yaml(testsets, output_format='yaml', output_file=None):
    output_dict = {}
    for row in testsets:
        # print(f'row: {row}')
        test_type, test_name, test_cmd = row
        if test_type not in output_dict:
            output_dict[test_type] = []
        testset = {"test_name": test_name, "test_cmd": test_cmd}
        output_dict[test_type].append(testset)
    # print(f'{output_dict=}')
    if output_file is None:
        return output_dict 
    else:
        with open(output_file, 'w') as file:
            if output_format == 'json':
                json.dump(output_dict, file)
            elif output_format == 'yaml':    
                yaml.dump(output_dict, file, default_flow_style=False)
        return 0


def parse_args():
    # Create the top-level parser
    parser = argparse.ArgumentParser(description='A command-line tool with subcommands')

    # Create subparser
    subparsers = parser.add_subparsers(dest='command', help='subcommand help')
    
    subparser = subparsers.add_parser('update-cluster-tables', help='create or update cluster related tables from yaml')
    subparser.add_argument('--yaml', type=str, required=True, help='the yaml file to update cluster related tables')
    
    subparser = subparsers.add_parser('update-machine-tables', help='create or update machine related tables from yaml')
    subparser.add_argument('--yaml', type=str, required=True, help='the yaml file to update machine related tables')

    subparser = subparsers.add_parser('update-testset-tables', help='create or update testset related tables from yaml')
    subparser.add_argument('--yaml', type=str, required=True, help='the yaml file to update testset related tables')

    subparser = subparsers.add_parser('query-cluster-metrics', help='query cluster metrics ')
    subparser.add_argument('--cluster', type=str, required=True, help='name of the cluster to query metrics')
    subparser.add_argument('-o', '--output-format', type=str, default='json', choices=["json", "yaml"], help='output format of the metrics')
    subparser.add_argument('-f', '--output-file', type=str, help='name of the output file to write metrics')
    
    subparser = subparsers.add_parser('query-machine-metrics', help='query machine metrics ')
    subparser.add_argument('--machine', type=str, required=True, help='name of the machine to query metrics')
    subparser.add_argument('-o', '--output-format', type=str, default='yaml', choices=["json", "yaml"], help='output format of the metrics')
    subparser.add_argument('-f', '--output-file', type=str, help='name of the output file to write metrics')
    
    subparser = subparsers.add_parser('query-testset', help='query testset ')
    subparser.add_argument('--testset', type=str, required=True, help='the testset to query')
    subparser.add_argument('--test-type', type=str, default=None, help='type of the testset to query')
    subparser.add_argument('--test-name', type=str, default=None, help='name of the testset to query')
    subparser.add_argument('-o', '--output-format', type=str, default='yaml', choices=["json", "yaml"], help='output format of the metrics')
    subparser.add_argument('-f', '--output-file', type=str, help='name of the output file to write metrics')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    if args.command == 'update-cluster-tables':
        create_or_update_cluster_tables_from_yaml(args.yaml)
    elif args.command == 'update-machine-tables':
        create_or_update_machine_tables_from_yaml(args.yaml)
    elif args.command == 'update-testset-tables':
        create_or_update_testset_tables_from_yaml(args.yaml)
    elif args.command == 'query-cluster-metricst':
        query_cluster_metrics(args.cluster, args.output_format, args.output_file)
    elif args.command == 'query-machine-metrics':
        query_machine_metrics(args.machine, args.output_format, args.output_file)
    elif args.command == 'query-testset':
        query_testset(args.testset, args.test_type, args.test_name, args.output_format, args.output_file)
    else:
        print("Unknown command")

if __name__ == "__main__":
    connect_to_mysql() 

