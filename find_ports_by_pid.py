import psutil

def find_ports_by_pid(pid):
    try:
        proc = psutil.Process(pid)
        connections = proc.connections(kind='inet')
        ports = [conn.laddr.port for conn in connections if conn.status == psutil.CONN_LISTEN]
        return ports
    except psutil.NoSuchProcess:
        return []

pid = 3176481
ports = find_ports_by_pid(pid)
if ports:
    print(f"Ports used by process {pid}: {ports}")
else:
    print(f"No ports found for process {pid} or the process is not listening on any port.")
