import threading
import os
import socket
try:
    from SimpleXMLRPCServer import (SimpleXMLRPCServer,
                                    SimpleXMLRPCRequestHandler)
    from SimpleHTTPServer import SimpleHTTPRequestHandler
except ImportError:
    # Python 3.x
    from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
    from http.server import SimpleHTTPRequestHandler

from multiprocessing.managers import BaseManager, BaseProxy


def get_authkey_bytes(authkey):
    if isinstance(authkey, bytes):
        return authkey
    else:
        return authkey.encode('utf-8')


class MultiprocessingInterface(BaseManager):
    """ A multiprocessing interface to the solver controller

    This object exports a controller instance proxy over the multiprocessing
    interface. Control actions can be performed by connecting to the interface
    and calling methods on the controller proxy instance """
    def __init__(self, address=None, authkey=None, try_next_port=False):
        authkey = get_authkey_bytes(authkey)
        BaseManager.__init__(self, address, authkey)
        self.authkey = authkey
        self.try_next_port = try_next_port

    def get_controller(self):
        return self.controller

    def start(self, controller):
        self.controller = controller
        self.register('get_controller', self.get_controller)
        if not self.try_next_port:
            self.get_server().serve_forever()
        host, port = self.address
        while self.try_next_port:
            try:
                BaseManager.__init__(self, (host, port), self.authkey)
                self.get_server().serve_forever()
                self.try_next_port = False
            except socket.error as e:
                try_next_port = False
                import errno
                if e.errno == errno.EADDRINUSE:
                    port += 1
                else:
                    raise


class MultiprocessingClient(BaseManager):
    """ A client for the multiprocessing interface

    Override the run() method to do appropriate actions on the proxy
    instance of the controller object or add an interface using the
    add_interface methods similar to the Controller.add_interface method """
    def __init__(self, address=None, authkey=None, serializer='pickle',
                 start=True):
        authkey = get_authkey_bytes(authkey)
        BaseManager.__init__(self, address, authkey, serializer)
        if start:
            self.start()

    def start(self, connect=True):
        self.interfaces = []

        # to work around a python caching bug
        # http://stackoverflow.com/questions/3649458/broken-pipe-when-using-python-multiprocessing-managers-basemanager-syncmanager
        if self.address in BaseProxy._address_to_local:
            del BaseProxy._address_to_local[self.address][0].connection

        self.register('get_controller')
        if connect:
            self.connect()
            self.controller = self.get_controller()
        self.run(self.controller)

    @staticmethod
    def is_available(address):
        try:
            socket.create_connection(address, 1).close()
            return True
        except socket.error:
            return False

    def run(self, controller):
        pass

    def add_interface(self, callable):
        """ This makes it act as substitute for the actual command_manager """
        thr = threading.Thread(target=callable, args=(self.controller,))
        thr.daemon = True
        thr.start()
        return thr


class CrossDomainXMLRPCRequestHandler(SimpleXMLRPCRequestHandler,
                                      SimpleHTTPRequestHandler):
    """ SimpleXMLRPCRequestHandler subclass which attempts to do CORS

    CORS is Cross-Origin-Resource-Sharing (http://www.w3.org/TR/cors/)
    which enables xml-rpc calls from a different domain than the xml-rpc server
    (such requests are otherwise denied)
    """
    def do_OPTIONS(self):
        """ Implement the CORS pre-flighted access for resources """
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-METHODS", "POST,GET,OPTIONS")
        # self.send_header("Access-Control-Max-Age", "60")
        self.send_header("Content-length", "0")
        self.end_headers()

    def do_GET(self):
        """ Handle http requests to serve html/image files only """
        print(self.path, self.translate_path(self.path))
        permitted_extensions = ['.html', '.png', '.svg', '.jpg', '.js']
        if not os.path.splitext(self.path)[1] in permitted_extensions:
            self.send_error(404, 'File Not Found/Allowed')
        else:
            SimpleHTTPRequestHandler.do_GET(self)

    def end_headers(self):
        """ End response header with adding Access-Control-Allow-Origin

        This is done to enable CORS request from all clients """
        self.send_header("Access-Control-Allow-Origin", "*")
        SimpleXMLRPCRequestHandler.end_headers(self)


class XMLRPCInterface(SimpleXMLRPCServer):
    """ An XML-RPC interface to the solver controller

    Currently cannot work with objects which cannot be marshalled
    (which is basically most custom classes, most importantly
    ParticleArray and numpy arrays) """
    def __init__(self, addr, requestHandler=CrossDomainXMLRPCRequestHandler,
                 logRequests=True, allow_none=True,
                 encoding=None, bind_and_activate=True):
        SimpleXMLRPCServer.__init__(self, addr, requestHandler, logRequests,
                                    allow_none, encoding, bind_and_activate)

    def start(self, controller):
        self.register_instance(controller, allow_dotted_names=False)
        self.register_introspection_functions()
        self.serve_forever()


class CommandlineInterface(object):
    """ command-line interface to the solver controller """
    def start(self, controller):
        while True:
            try:
                try:
                    inp = raw_input('pysph[%d]>>> ' % controller.get('count'))
                except NameError:
                    inp = input('pysph[%d]>>> ' % controller.get('count'))
                cmd = inp.strip().split()
                try:
                    cmd, args = cmd[0], cmd[1:]
                except Exception as e:
                    print('Invalid command')
                    self.help()
                    continue
                args2 = []
                for arg in args:
                    try:
                        arg = eval(arg)
                    except:
                        pass
                    finally:
                        args2.append(arg)

                if cmd == 'p' or cmd == 'pause':
                    controller.pause_on_next()
                elif cmd == 'c' or cmd == 'cont':
                    controller.cont()
                elif cmd == 'g' or cmd == 'get':
                    print(controller.get(args[0]))
                elif cmd == 's' or cmd == 'set':
                    print(controller.set(args[0], args2[1]))
                elif cmd == 'q' or cmd == 'quit':
                    break
                else:
                    print(getattr(controller, cmd)(*args2))
            except Exception as e:
                self.help()
                print(e)

    def help(self):
        print('''Valid commands are:
    p | pause
    c | cont
    g | get <name>
    s | set <name> <value>
    q | quit -- quit commandline interface (solver keeps running)''')
