from pymodbus.client import ModbusTcpClient
 
# Define the Modbus server address and port

MODBUS_SERVER_IP = '192.168.1.200'

MODBUS_SERVER_PORT = 502
 
# Create a Modbus TCP client

class PumpControl():
    def __init__(self, server_ip=MODBUS_SERVER_IP, _port=MODBUS_SERVER_PORT):
        self.client = ModbusTcpClient(server_ip, port=_port)
        print(f"Initializing Modbus server ip: {server_ip}, port: {_port}")
        self.client.connect()
        print(f"Connected to Modbus server")

    def config_gripper(self, cmd):
        if cmd == 1: 
            state_ = 'open'
        resp = self.client.write_coil(257, cmd)
        return resp
    
    def get_state(self):
        return 

    def release_hardware(self):
        print("Releasing hardware")
        return