''' Implement infrastructure for the solver to add various interfaces '''

from functools import wraps
import threading
try:
    from thread import LockType
except ImportError:
    from _thread import LockType
from pysph.base.particle_array import ParticleArray

import logging
logger = logging.getLogger(__name__)

class DummyComm(object):
    ''' A dummy MPI.Comm implementation as placeholder for for serial runs '''
    def Get_size(self):
        ''' return the size of the comm (1) '''
        return 1

    def Get_rank(self):
        ''' return the rank of the process (0) '''
        return 0

    def send(self, data, pid):
        ''' dummy send implementation '''
        self.data = data

    def recv(self, pid):
        ''' dummy recv implementation '''
        data = self.data
        del self.data
        return data

    def bcast(self, data):
        ''' bcast (broadcast) implementation for serial run '''
        return data

    def gather(self, data):
        ''' gather implementation for serial run '''
        return [data]

def synchronized(lock_or_func):
    ''' decorator for synchronized (thread safe) function

    Usage:

    - sync_func = synchronized(lock)(func) # sync with an existing lock

    - sync_func = synchronized(func) # sync with a new private lock
    '''
    if isinstance(lock_or_func, LockType):
        lock = lock_or_func
        def synchronized_inner(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                with lock:
                    return func(*args, **kwargs)
            return wrapped
        return synchronized_inner
    else:
        func = lock_or_func
        lock = threading.Lock()
        return synchronized(lock)(func)

def wrap_dispatcher(obj, meth, *args2, **kwargs2):
    @wraps(meth)
    def wrapped(*args, **kwargs):
        kw = {}
        kw.update(kwargs2)
        kw.update(kwargs)
        return meth(obj.block, *(args2+args), **kw)
    return wrapped

class Controller(object):
    ''' Controller class acts a a proxy to control the solver

    This is passed as an argument to the interface

    **Methods available**:

    - get -- get the value of a solver parameter
    - set -- set the value of a solver parameter
    - get_result -- return result of a queued command
    - pause_on_next -- pause solver thread on next iteration
    - wait -- wait (block) calling thread till solver is paused
      (call after `pause_on_next`)
    - cont -- continue solver thread (call after `pause_on_next`)

    Various other methods are also available as listed in
    :data:`CommandManager.dispatch_dict` which perform different functions.

    - The methods in CommandManager.active_methods do their operation and return
      the result (if any) immediately
    - The methods in CommandManager.lazy_methods do their later when solver
      thread is available and return a task-id. The result of the task can be
      obtained later using the blocking call `get_result()` which waits till
      result is available and returns the result.
      The availability of the result can be checked using the lock returned
      by `get_task_lock()` method

    FIXME: wait/cont currently do not work in parallel

    '''
    def __init__(self, command_manager, block=True):
        super(Controller, self).__init__()
        self.__command_manager = command_manager
        self.daemon = True
        self.block = block
        self._set_methods()

    def _set_methods(self):
        for prop in self.__command_manager.solver_props:
            setattr(self, 'get_'+prop, wrap_dispatcher(self, self.__command_manager.dispatch, 'get', prop))
            setattr(self, 'set_'+prop, wrap_dispatcher(self, self.__command_manager.dispatch, 'set', prop))

        for meth in self.__command_manager.solver_methods:
            setattr(self, meth, wrap_dispatcher(self, self.__command_manager.dispatch, meth))

        for meth in self.__command_manager.lazy_methods:
            setattr(self, meth, wrap_dispatcher(self, self.__command_manager.dispatch, meth))

        for meth in self.__command_manager.active_methods:
            setattr(self, meth, wrap_dispatcher(self, self.__command_manager.dispatch, meth))

    def get(self, name):
        ''' get a solver property; returns immediately '''
        return self.__command_manager.dispatch(self.block, 'get', name)

    def set(self, name, value):
        ''' set a solver property; returns immediately; '''
        return self.__command_manager.dispatch(self.block, 'set', name, value)

    def pause_on_next(self):
        ''' pause the solver thread on next iteration '''
        return self.__command_manager.pause_on_next()

    def wait(self):
        ''' block the calling thread until the solver thread pauses

        call this only after calling the `pause_on_next` method to tell
        the controller to pause the solver thread'''
        self.__command_manager.wait()
        return True

    def get_prop_names(self):
        return list(self.__command_manager.solver_props)

    def cont(self):
        ''' continue solver thread after it has been paused by `pause_on_next`

        call this only after calling the `pause_on_next` method '''
        return self.__command_manager.cont()

    def get_result(self, task_id):
        ''' get the result of a previously queued command '''
        return self.__command_manager.get_result(task_id)

    def set_blocking(self, block):
        ''' set the blocking mode to True/False

        In blocking mode (block=True) all methods other than getting of
        solver properties block until the command is executed by the solver
        and return the results. The blocking time can vary depending on the
        time taken by solver per iteration and the command_interval
        In non-blocking mode, these methods queue the command for later
        and return a string corresponding to the task_id of the operation.
        The result can be later obtained by a (blocking) call to get_result
        with the task_id as argument
        '''
        if block != self.block:
            self.block = block
            self._set_methods()
        return self.block

    def get_blocking(self):
        ''' get the blocking mode ( True/False ) '''
        return self.block

    def ping(self):
        return True

def on_root_proc(f):
    ''' run the decorated function only on the root proc '''
    @wraps(f)
    def wrapper(self, *args, **kwds):
        if self.comm.Get_rank()==0:
            return f(self, *args, **kwds)
    return wrapper

def in_parallel(f):
    ''' return a list of results of running decorated function on all procs '''
    @wraps(f)
    def wrapper(self, *args, **kwds):
        return self.comm.gather(f(self, *args, **kwds))
    return wrapper


class CommandManager(object):
    ''' Class to manage and synchronize commands from various Controllers '''

    solver_props = set(('t', 'tf', 'dt', 'count', 'pfreq', 'fname',
                'detailed_output', 'output_directory', 'command_interval'))

    solver_methods = set(('dump_output',))

    lazy_methods = set(('get_particle_array_names', 'get_named_particle_array',
                'get_particle_array_combined', 'get_particle_array_from_procs'))

    active_methods = set(('get_status', 'get_task_lock', 'set_log_level'))

    def __init__(self, solver, comm=None):
        if comm is not None:
            self.comm = comm
            self.rank = comm.Get_rank()
        else:
            try:
                self.comm = solver.particles.cell_manager.parallel_controller.comm
            except AttributeError:
                self.comm = DummyComm()
            self.rank = 0
        logger.debug('CommandManager: using comm: %s'%self.comm)
        self.solver = solver
        self.interfaces = []
        self.func_dict = {}
        self.rlock = threading.RLock()
        self.res_lock = threading.Lock()
        self.plock = threading.Condition()
        self.qlock = threading.Condition() # queue lock
        self.queue = []
        self.queue_dict = {}
        self.queue_lock_map = {}
        self.results = {}
        self.pause = set([])

    @on_root_proc
    def add_interface(self, callable, block=True):
        ''' Add a callable interface to the controller

        The callable must accept an Controller instance argument.
        The callable is called in a new thread of its own and it can
        do various actions with methods defined on the Controller
        instance passed to it
        The new created thread is set to daemon mode and returned
        '''
        logger.debug('adding_interface: %s'%callable)
        control = Controller(self, block)
        thr = threading.Thread(target=callable, args=(control,))
        thr.daemon = True
        thr.start()
        return thr

    def add_function(self, callable, interval=1):
        ''' add a function to to be called every `interval` iterations '''
        l = self.func_dict[interval] = self.func_dict.get(interval, [])
        l.append(callable)

    def execute_commands(self, solver):
        ''' called by the solver after each timestep '''
        # TODO: first synchronize all the controllers in different processes
        # using mpi
        self.sync_commands()
        with self.qlock:
            self.run_queued_commands()
        if self.rank == 0:
            logger.debug('control handler: count=%d'%solver.count)

        for interval in self.func_dict:
            if solver.count%interval == 0:
                for func in self.func_dict[interval]:
                    func(solver)

        self.wait_for_cmd()

    def wait_for_cmd(self):
        ''' wait for command from any interface '''
        with self.qlock:
            while self.pause:
                with self.plock:
                    self.plock.notify_all()
                self.qlock.wait()
                self.run_queued_commands()

    def sync_commands(self):
        ''' send the pending commands to all the procs in parallel run '''
        self.queue_dict, self.queue, self.pause = self.comm.bcast((self.queue_dict, self.queue, self.pause))


    def run_queued_commands(self):
        while self.queue:
            lock_id = self.queue.pop(0)
            meth, args, kwargs = self.queue_dict[lock_id]
            with self.res_lock:
                try:
                    self.results[lock_id] = self.run_command(meth, args, kwargs)
                finally:
                    del self.queue_dict[lock_id]
                    if self.comm.Get_rank()==0:
                        self.queue_lock_map[lock_id].release()

    def run_command(self, cmd, args=[], kwargs={}):
        res =  self.dispatch_dict[cmd](self, *args, **kwargs)
        logger.debug('controller: running_command: %s %s %s %s'%(
                                                cmd, args, kwargs, res))
        return res

    def pause_on_next(self):
        ''' pause and wait for command on the next control interval '''
        if self.comm.Get_size() > 1:
            logger.debug('pause/continue not yet supported in parallel runs')
            return False
        with self.plock:
            self.pause.add(threading.current_thread().ident)
            self.plock.notify()
        return True

    def wait(self):
        with self.plock:
            self.plock.wait()

    def cont(self):
        ''' continue after a pause command '''
        if self.comm.Get_size() > 1:
            logger.debug('pause/continue noy yet supported in parallel runs')
            return
        with self.plock:
            self.pause.remove(threading.current_thread().ident)
            self.plock.notify()
            with self.qlock:
                self.qlock.notify_all()

    def get_result(self, lock_id):
        ''' get the result of a previously queued command '''
        lock_id = int(lock_id)
        lock = self.queue_lock_map[lock_id]
        with lock:
            with self.res_lock:
                ret = self.results[lock_id]
                del self.results[lock_id]
                del self.queue_lock_map[lock_id]
            return ret

    def get_task_lock(self, lock_id):
        ''' get the Lock instance associated with a command '''
        return self.queue_lock_map[int(lock_id)]

    def get_prop(self, name):
        ''' get a solver property '''
        return getattr(self.solver, name)

    def set_prop(self, name, value):
        ''' set a solver property '''
        return setattr(self.solver, name, value)

    def solver_method(self, name, *args, **kwargs):
        ''' execute a method on the solver '''
        ret = getattr(self.solver, name)(*args, **kwargs)
        ret = self.comm.gather(ret)
        return ret

    def get_particle_array_names(self):
        ''' get the names of the particle arrays '''
        return [pa.name for pa in self.solver.particles]

    def get_named_particle_array(self, name, props=None):
        for pa in self.solver.particles:
            if pa.name == name:
                if props:
                    return [getattr(pa, p) for p in props if hasattr(pa, p)]
                else:
                    return pa

    def get_particle_array_index(self, name):
        ''' get the index of the named particle array '''
        for i,pa in enumerate(self.solver.particles):
            if pa.name == name:
                return i

    def get_particle_array_from_procs(self, idx, procs=None):
        ''' get particle array at index from all processes

        specifying processes is currently not implemented
        '''
        if procs is None:
            procs = list(range(self.comm.size))
        pa = self.solver.particles[idx]
        pas = self.comm.gather(pa)
        return pas

    def get_particle_array_combined(self, idx, procs=None):
        ''' get a single particle array with combined data from all procs

        specifying processes is currently not implemented
        '''
        if procs is None:
            procs = list(range(self.comm.size))
        pa = self.solver.particles[idx]
        pas = self.comm.gather(pa)
        pa = ParticleArray(name=pa.name)
        for p in pas:
            pa.append_parray(p)
        return pa

    def get_status(self):
        ''' get the status of the controller '''
        return 'commands queued: %d'%len(self.queue)

    def set_log_level(self, level):
        ''' set the logging level '''
        logger.setLevel(level)

    dispatch_dict = {'get':get_prop, 'set':set_prop}

    for meth in solver_methods:
        dispatch_dict[meth] = solver_method

    for meth in lazy_methods:
        dispatch_dict[meth] = locals()[meth]

    for meth in active_methods:
        dispatch_dict[meth] = locals()[meth]

    @synchronized
    def dispatch(self, block, meth, *args, **kwargs):
        ''' execute/queue a command with specified arguments '''
        if meth in self.dispatch_dict:
            if meth=='get' or meth=='set':
                prop = args[0]
                if prop not in self.solver_props:
                    raise RuntimeError('Invalid dispatch on method: %s with '
                                       'non-existant property: %s '%(meth,prop))
            if block or meth=='get' or meth in self.active_methods:
                logger.debug('controller: immediate dispatch(): %s %s %s'%(
                            meth, args, kwargs))
                return self.dispatch_dict[meth](self, *args, **kwargs)
            else:
                lock = threading.Lock()
                lock.acquire()
                lock_id = id(lock)
                with self.qlock:
                    self.queue_lock_map[lock_id] = lock
                    self.queue_dict[lock_id] = (meth, args, kwargs)
                    self.queue.append(lock_id)
                logger.debug('controller: dispatch(%d): %s %s %s'%(
                            lock_id, meth, args, kwargs))
                return str(lock_id)
        else:
            raise RuntimeError('Invalid dispatch on method: '+meth)
