import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Global_Time():
    """
    Class representing global time.

    Attributes:
        t (float): Current time.
        dt (float): Time step size.

    Methods:
        step(): Advances the time by one time step.
        __call__(): Returns the current time.
    """

    def __init__(self, dt: float):
        self.t = 0
        self.dt = dt

    def step(self):
        self.t += self.dt

    def __call__(self):
        return self.t



class StringModel():
    """
    A class representing a string model.

    Parameters:
    - x (np.ndarray): The spatial coordinates of the string.
    - y0 (np.ndarray): The initial shape of the string.
    - dt (float): The time step size.
    - c (float): The wave speed.
    - L (float): The length of the string.
    - b (float, optional): The damping coefficient. Defaults to 0.

    Methods:
    - set_sine_disturbancer(a, d, f='resonant'): Sets the sine wave disturbance parameters.
    - set_hammer_attack(vel, d, att_time, size): Sets the hammer attack disturbance parameters.
    - set_noise_disturbancer(noise): Sets the noise disturbance parameters.
    - set_initial_shape(y0): Sets the initial shape of the string.
    - step(): Performs a time step in the simulation.
    - get_y(): Returns the current shape of the string.
    - test(iter=1000, plot=True): Runs a test simulation of the string model.
    - reset(): Resets the string model to its initial state.
    """
    def __init__(self, 
                 x: np.ndarray,
                 y0: np.ndarray,
                 dt: float,
                 c: float,
                 L: float,
                 b: float = 0):
        
        self.x = np.copy(x)
        self.c = c
        self.L = L
        self.b = b
        
        # ICs
        self.set_initial_shape(y0)
        
        # set timer
        self.dt = dt
        self.time = Global_Time(dt)

        # set physical disturbance
        self.sine_disturbancer = None
        self.hammer_attack = None
        self.noise_disturbancer = None

    def pad_array(self, arr):
        return np.concatenate((arr[:1], arr, arr[-1:]))
    
    def set_sine_disturbancer(self,
                              a: float,
                              d: float,
                              f: float | str = 'resonant'):
        """
        Sets the sine disturbancer for the string model.

        Parameters:
        - a: The amplitude of the disturbance.
        - d: The position of the disturbance (ranging from 0 to 1).
        - f: The frequency of the disturbance. Defaults to 'resonant' to generate a resonant frequency.
        """
        self.sine_disturbancer = SineDisturbulancer(self, a, d, f)
    
    def set_hammer_attack(self, vel, d, att_time, size):
        """
        Sets the hammer attack for the StringModel.

        Parameters:
        - vel (float): The velocity of the hammer attack.
        - d (float): The distance of the hammer attack (ranging from 0 to 1)
        - att_time (float): The attack time of the hammer attack.
        - size (float): The size of the hammer attack.
        """
        self.hammer_attack = HammerAttack(self, vel, d, att_time, size)

    def set_noise_disturbancer(self, noise):
        """
        Sets the noise and initializes the noise disturbancer.

        Parameters:
            noise (float): The noise value to be set.
        """
        self.noise = noise
        self.noise_disturbancer = NoiseDisturbulancer(self, noise)

    def set_initial_shape(self, y0):
        self.y = self.pad_array(y0)
        self.y0 = self.pad_array(y0)
        self.y_prev = np.copy(self.y0)
    
    def step(self):
            """
            Perform a single time step in the simulation.

            This method updates the state of the string model by advancing it by one time step.
            It calculates the new values of the string displacement based on the previous values,
            the boundary conditions, and any applied disturbances.
            """
            cc = (self.c * self.dt / np.gradient(self.x)) ** 2
            bb = self.b * self.dt / 2

            tmp = np.copy(self.y)
            self.y[1:-1] = 2 * self.y[1:-1] + (bb - 1) * self.y_prev[1:-1] + cc * (self.y[2:] - 2 * self.y[1:-1] + self.y[:-2])
            self.y[1:-1] /= (1 + bb)

            # [:] to asure that self.y_prev is the same id
            self.y_prev[:] = tmp[:]

            
            if self.sine_disturbancer is not None:
                self.sine_disturbancer.step()
            
            if self.hammer_attack is not None:
                self.hammer_attack.step()

            if self.noise_disturbancer is not None:
                self.noise_disturbancer.step()
            
            # BCs
            self.y[[0,1,-2,-1]] = self.y0[[0,1,-2,-1]]
            
            # update time
            self.time.step()

    def get_y(self):
        return self.y[1:-1]
    
    def test(self, iter=1000, plot=True):
        print('Testing...')
        for i in tqdm(range(iter)):
            self.step()
            if np.any(np.abs(self.y[1:-1] > 1e10)):
                raise ValueError('Model diverged! Please adjust parameters.')
            if i % 100 == 0:
                # print(i)
                plt.plot(self.x, self.y[1:-1])
        
        self.reset()

        if plot:
            plt.show()

    def reset(self):
        self.y[:] = self.y0[:]
        self.y_prev[:] = self.y0[:]
        self.time.t = 0

        if self.sine_disturbancer is not None:
            self.sine_disturbancer.reset()
        
        if self.hammer_attack is not None:
            self.hammer_attack.reset()

        if self.noise_disturbancer is not None:
            self.noise_disturbancer.reset()
    

class SineDisturbulancer():
    """
    A class representing a sine wave disturbance generator for a string model.

    Args:
        parent (StringModel): The parent string model object.
        a (float): The amplitude of the disturbance.
        d (float): The position of the disturbance (ranging from 0 to 1).
        f (float | str): The frequency of the disturbance. Can be a float or "resonant" for generating resonant frequency.

    Methods:
        step(): Advances the disturbance by one step.
        reset(): Resets the disturbance.

    """

    def __init__(self,
                 parent: StringModel,
                 a: float,
                 d: float,
                 f: float | str):
        self.parent = parent

        self.a = a
        self.d = int(self.parent.y[1:-1].shape[0] * d)

        if f == 'resonant':
            # resonant frequency
            _lambda = d * self.parent.L * 4
            self.f = self.parent.c / _lambda
        elif type(self.f) == float:
            self.f = f
        else:
            raise ValueError('Invalid frequency, expected float or "resonant" for generating resonant frequency, got {} instead', self.f)
        
    def step(self):
        """
        Perform a step in the simulation.

        This method updates the value of `self.y` at index `self.d` based on the current time.
        """
        self.parent.y[self.d] = self.a * np.sin(2 * np.pi * self.f * self.parent.time())

    def reset(self):
        pass


class HammerAttack():
    def __init__(self, parent: StringModel, vel: float, d: float, att_time: float, size: float):
        """
        Initializes a HammerAttack object.

        Parameters:
        - parent (StringModel): The parent StringModel object.
        - vel (float): The velocity of the attack.
        - d (float): The position of the attack (ranging from 0 to 1).
        - att_time (float): The time at which the attack occurs.
        - size (float): The size of the attack.
        """
        self.parent = parent
        self.vel = vel / size
        self.d = int(self.parent.y[1:-1].shape[0] * d) # position
        self.att_time = att_time
        self.size = size if size >= 3 else 3
        self.attdist = self.vel * self.parent.dt
        self.attacked = False

    def step(self):
        """
        Performs a step in the HammerAttack.
        """
        if self.parent.time() >= self.att_time and not self.attacked:
            for s in range(-self.size//2, self.size//2):
                self.parent.y_prev[self.d+s] += self.attdist * np.sin(np.pi * s / self.size - np.pi / 2)
            self.attacked = True

    def reset(self):
        self.attacked = False

class NoiseDisturbulancer():
    """
    A class that adds noise to a StringModel object.

    Args:
        parent (StringModel): The parent StringModel object.
        noise (float): The initial noise value.

    Methods:
        step(): Updates the parent StringModel object by adding noise.
        reset(): Resets the noise value to its initial value.
    """

    def __init__(self, parent: StringModel, noise: float):
        self.parent = parent
        self.init_noise = noise
        self.noise = noise

    def step(self):
        """
        Updates the parent StringModel object by adding noise.
        """
        self.parent.y += np.random.normal(0, self.noise, self.parent.y.shape)
        bb = self.parent.b * self.parent.dt / 2
        self.noise /= (1 + bb)

    def reset(self):
        self.noise = self.init_noise

class bowDisturbulancer():
    def __init__(self):
        raise NotImplementedError()