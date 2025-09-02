import socket
import urllib.parse
import orjson
import msgpack
# import torch
import numpy as np
import select
import threading
import queue
import time
import inspect

class DataPublisher:
    """
    Manages data publishing to a specified target URL.

    Methods
    -------
    publish(data: dict)
        Sends data to the target URL if publishing is enabled.
    """

    SUPPORTED_SCHEMES = {'unix', 'tcp', 'udp'}

    def __init__(
        self,
        target_url: str = 'udp://localhost:9870',
        encoding: str = 'msgpack',
        broadcast: bool = False,
        enable: bool = True,
        thread: bool = True,
        save_to_file: bool = False,
        save_file_path: str = None,
        **socket_kwargs,
    ):
        """
        Initialize DataPublisher with connection and encoding details.

        Parameters
        ----------
        target_url : str
            URL to which data will be sent.
        encoding : str
            Encoding for sending data, default is 'json'.
        broadcast : bool
            If True, data is broadcasted; defaults to True.
        enable : bool
            If False, publishing is inactive; defaults to False.
        thread : bool
            If True, publishing is done in a separate thread; defaults to True.
        socket_kwargs : 
            Additional keyword arguments for the socket.
        """
        self.enable = enable
        self.url = urllib.parse.urlparse(target_url)

        # Validate scheme
        if self.url.scheme not in self.SUPPORTED_SCHEMES:
            raise ValueError(f"Unsupported scheme in URL: {target_url}")

        # Set socket family and type
        family = self._get_socket_family()
        self.is_tcp = self.url.scheme == 'tcp'
        if self.is_tcp:
            socket_type = socket.SOCK_STREAM
            self.connected = False
        else:
            socket_type = socket.SOCK_DGRAM
            # self.connected = True
        # socket_type = socket.SOCK_STREAM if self.is_tcp else socket.SOCK_DGRAM
        self.socket = socket.socket(family, socket_type, **socket_kwargs)
        if broadcast and not self.is_tcp:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.hostname = '<broadcast>' if broadcast else self.url.hostname

        self._setup_encoding(encoding)

        if thread:
            self.data_queue = queue.Queue()
            self.stop_event = threading.Event()
            self.publisher_thread = threading.Thread(target=self._publisher_thread_func, daemon=True)
            self.publisher_thread.start()
            self.publish = self._publish_continuously
            self.__del__ = self._stop
        else:
            self.publish = self._send_data

        self.save_to_file: bool = save_to_file
        if self.save_to_file:
            if save_file_path is None:
                raise ValueError("save_file_path must be provided if save_to_file is True")
            self.save_file_path = save_file_path

    def _get_socket_family(self):
        """Determine and return the appropriate socket family based on the URL."""
        if 'unix' in self.url.scheme:
            return socket.AF_UNIX
        return socket.AF_INET6 if ':' in (self.url.hostname or '') else socket.AF_INET

    def _setup_encoding(self, encoding_type):
        """Configure the data encoding method based on the provided encoding_type."""
        encodings = {
            'raw': lambda data: data,  # raw/bytes
            'utf-8': lambda data: data.encode('utf-8'),
            'msgpack': lambda data: msgpack.packb(data, use_single_float=False, use_bin_type=True),
            'json': lambda data: orjson.dumps(data),
        }
        if encoding_type not in encodings:
            raise ValueError(f'Invalid encoding: {encoding_type}')
        self.encode = encodings[encoding_type]
        self.should_encode = encoding_type != 'raw'

    def _send_data(self, data: dict):
        if self.enable:
            converted_data = convert_to_python_builtin_types(data) if self.should_encode else data
            encoded_data = self.encode(converted_data)
            try:
                if self.is_tcp:
                    try:
                        if self.connected:
                            self.socket.sendall(encoded_data)
                        if not self.connected:
                            self.socket.connect((self.hostname, self.url.port))
                            self.connected = True
                            self.socket.sendall(encoded_data)
                    except (BrokenPipeError, ConnectionResetError):
                        # print file and line number in orange, do not use trace back, print the file name and line number
                        print(f"\033[93m{__file__}:{inspect.currentframe().f_lineno} [Publisher] Connection lost. Attempting to reconnect...\033[0m")
                        try:
                            self.socket.close()
                            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            self.socket.connect((self.hostname, self.url.port))
                            self.socket.sendall(encoded_data)
                            print("[Publisher] Reconnected successfully.")
                        except Exception as reconnect_error:
                            print(f"\033[93m{__file__}:{inspect.currentframe().f_lineno} [Publisher] Reconnection failed: {reconnect_error}\033[0m")
                else:
                    self.socket.sendto(encoded_data, (self.hostname, self.url.port))
                    pass
                if self.save_to_file:
                    with open(self.save_file_path, 'ab') as f:
                        f.write(encoded_data)
            except Exception as e:
                print(f"\033[93m{__file__}:{inspect.currentframe().f_lineno} [Publisher] Failed to send data: {e}\033[0m")

    def _publish_continuously(self, data: dict):
        self.data_queue.put(data, block=False)

    def _publisher_thread_func(self):
        """Function run by the publisher thread to send data from the queue."""
        while not self.stop_event.is_set():
            try:
                data = self.data_queue.get(timeout=0.1)  # Wait for data with timeout
                self._send_data(data)
            except queue.Empty:
                continue

    def _stop(self):
        """Stops the publishing thread."""
        self.stop_event.set()
        self.publisher_thread.join()
        self.socket.close()


class DataReceiver:
    def __init__(self, port=9870, decoding="msgpack", broadcast=False, enable=True, protocol="udp"):
        self.protocol = protocol
        self.enable = enable

        if protocol == "tcp":
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('', port))
            self.socket.listen(1)
            self.client_socket = None
            self.client_address = None
        else:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            if broadcast:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.socket.bind(('<broadcast>' if broadcast else '', port))
            self.socket.setblocking(False)
        self.decodings = {
            "raw": lambda data: data,  # raw/bytes
            "utf-8": lambda data: data.decode("utf-8"),
            "msgpack": lambda data: msgpack.unpackb(data),
            "json": lambda data: orjson.loads(data),
        }
        self.decoding = decoding.lower()
        if protocol == "tcp" and self.decoding == "msgpack":
            self.unpacker = msgpack.Unpacker(raw=False)
        else:
            self.unpacker = None
        if self.decoding not in self.decodings:
            raise ValueError(f"Invalid decoding: {self.decoding}")
        self._decode = self.decodings[self.decoding]

        self._running = True
        self.data_id = -1
        self.data = None
        self.address = None

    def decode(self, data):
        if self.unpacker:
            self.unpacker.feed(data)
            for unpacked in self.unpacker:
                return unpacked 
        else:
            return self._decode(data)

    def _accept_tcp_connection(self):
        if self.client_socket is None:
            # print(f"Waiting for TCP connection on {self.socket.getsockname()}...")
            self.client_socket, self.client_address = self.socket.accept()
            self.client_socket.setblocking(False)
            self.address = self.client_address
            print(f"\033[92m{__file__}:{inspect.currentframe().f_lineno} [Publisher] TCP connection accepted from {self.client_address}\033[0m")

    def receive(self, timeout=0.1, buffer_size=1024 * 1024 * 8):
        if self.protocol == "tcp":
            self._accept_tcp_connection()
            ready = select.select([self.client_socket], [], [], timeout)
            if ready[0]:
                data = self.client_socket.recv(buffer_size)
                if not data:
                    self.client_socket.close()
                    self.client_socket = None
                    return None, None
                self.data = self.decode(data)
                self.data_id += 1
                return self.data, self.client_address
        else:
            ready = select.select([self.socket], [], [], timeout)
            if ready[0]:
                data, self.address = self.socket.recvfrom(buffer_size)
                self.data = self.decode(data)
                self.data_id += 1
                return self.data, self.address
        return None, None

    def receive_continuously(self, timeout=0.1, buffer_size=1024 * 1024 * 8):
        def _receive_loop():
            while self._running:
                if self.enable:
                    self.receive(timeout=timeout, buffer_size=buffer_size)
                else:
                    time.sleep(timeout)
        thread = threading.Thread(target=_receive_loop, daemon=True)
        thread.start()

    def stop(self):
        self._running = False
        if self.protocol == "tcp" and self.client_socket:
            self.client_socket.close()
        self.socket.close()


def convert_to_python_builtin_types(nested_data: dict):
    """
    Converts nested data (including tensors and arrays) to built-in types.

    Parameters
    ----------
    nested_data : dict
        Data to be converted.

    Returns
    -------
    dict
        Data converted to Python built-in types.
    """
    converted_data = {}
    for key, value in nested_data.items():
        if isinstance(value, dict):
            converted_data[key] = convert_to_python_builtin_types(value)
        elif hasattr(value, 'tolist'):
            converted_data[key] = value.tolist()
        else:
            converted_data[key] = value
    return converted_data


def unpack_data_from_file(file_path: str, decoding: str = "msgpack"):
    """
    Unpacks data from a file using the specified decoding method.

    Parameters
    ----------
    file_path : str
        Path to the file containing packed data.
    decoding : str
        Decoding method to use (e.g., 'msgpack', 'json').

    Returns
    -------
    list
        List of unpacked data.
    """
    with open(file_path, 'rb') as f:
        if decoding == "msgpack":
            return [msg for msg in msgpack.Unpacker(f, raw=False)]
        # elif decoding == "json":
        #     return orjson.loads(f.read())
        else:
            raise ValueError(f"Unsupported decoding: {decoding}")



# Example usage (just change protocol to "tcp" in both publisher and receiver)
if __name__ == "__main__":

    def test_data(i):
        return {
            "id": i,
            "sensor_id": np.random.randint(0, 10),
            "temperature": 25.5,
            "time": time.time(),
        }

    def test_tcp():
        # PRINT IN GREEN COLOR IN TERMINAL
        print("\033[92m" + "Starting publisher and receiver with TCP." + "\033[0m")

        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, mode='w+b')
        path = tmp.name

        # Publisher: TCP
        publisher = DataPublisher(
            target_url="tcp://localhost:9872", encoding="msgpack", broadcast=False, thread=True,
            save_to_file=True, save_file_path=path
            )
        
        # Receiver: TCP
        receiver = DataReceiver(port=9872, decoding="msgpack", protocol="tcp")
        receiver.receive_continuously()

        for i in range(10):
            publisher.publish(test_data(i))
            time.sleep(1e-3)  # add a small delay
            print(f"[Receiver] {receiver.data_id}: {receiver.data}")
        receiver.stop()
        print("Stopped.")

        # Load the data from the file
        with open(path, "rb") as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            results = [msg for msg in unpacker]
        print(f"Data saved to {path}. Loaded {len(results)} messages from the file.")
        print(f"All messages: {results}")


    def test_udp():
        # PRINT IN GREEN COLOR IN TERMINAL
        print("\033[92m" + "Starting publisher and receiver with UDP." + "\033[0m")
        # Create a publisher instance
        publisher = DataPublisher(
            target_url="udp://localhost:9870", encoding="msgpack",broadcast=True,thread=True)

        # Create a receiver instance
        num_receivers = 2
        receivers = []
        for i in range(num_receivers):
            receiver = DataReceiver(port=9870, decoding="msgpack",broadcast=True)
            # Start continuous receiving in a thread
            receiver.receive_continuously()     
            receivers.append(receiver)

        # Send data multiple times with a delay
        for i in range(10):
            publisher.publish(test_data(i))
            time.sleep(1e-3)  # add a small delay
            for k,receiver in enumerate(receivers):
                print(f"receiver [{k}] {receiver.data_id}, {receiver.address}: {receiver.data}")

        # Stop continuous receiving after a while
        receiver.stop()

        print("Publisher and Receiver have stopped.")
    
    
    test_tcp()
    # test_udp()




