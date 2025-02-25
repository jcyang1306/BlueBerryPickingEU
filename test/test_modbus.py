from pymodbus.client import ModbusTcpClient
 
# Define the Modbus server address and port

MODBUS_SERVER_IP = '192.168.1.200'

MODBUS_SERVER_PORT = 502
 
# Create a Modbus TCP client

client = ModbusTcpClient(MODBUS_SERVER_IP, port=MODBUS_SERVER_PORT)
 
# Connect to the Modbus server

if client.connect():

    print("Connected to Modbus server")
 
    # Example: Write to a holding register (address 0, value 123)

    cmd = 0
    while True:
        cmd = not cmd
        blk = input("press any key to switch state")
        print(f' cmd: {cmd}')
        write_response = client.write_coil(257, cmd)

    write_response = client.write_coil(256, 0)

    if write_response.isError():

        print("Error writing to holding register:", write_response)

    else:

        print("Successfully wrote value to holding register")
 
    # Close the connection
    client.close()

else:

    print("Failed to connect to Modbus server")
 