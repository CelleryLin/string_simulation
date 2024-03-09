 # String Simulation
 
 ## Define the model

### Modeling
From the wave equation with damping
$$
\frac{\partial^2 y(x,t)}{\partial t^2}+b \frac{\partial y(x,t)}{\partial t}=c^2 \frac{\partial^2 y(x,t)}{\partial x^2}
$$
where
- $y(x,t)$: displacement in y direction
- $t$: time
- $x$: displacement in x direction
- $b$: damping coefficient
- $c$: wave speed

using finite difference method (FDM), we can discretize the equation as
$$
u_i^{n+1}=\left(1+\frac{1}{2} b \Delta t\right)^{-1}\left(\left(\frac{1}{2} b \Delta t-1\right) u_i^{n-1}+2 u_i^n+C^2\left(u_{i+1}^n-2 u_i^n+u_{i-1}^n\right)\right).
$$
(explicit method)

And the stability criterion is:
$$
\frac{c\Delta t}{\Delta x} \leq 1
$$

### Initial Condition (ICs)
define a initial condition $y0$ of a string, can be any shapes.

### Boundary Condition (BCs)
we define the edges of the string as fixed, i.e. the displacement is zero at the $y(x=0)$ and  $y(x=L)$ all the time.
you can adjust the BCs in the `StringModel.step`.

```python
class StringModel():

    #...

    def step(self):

        #...

        # BCs
        self.y[[0,1,-2,-1]] = self.y0[[0,1,-2,-1]] # adjust here!


```

### Disturbancer
you can add 3 types of disturbancer to the system
- "sine disturbancer": make a point in the string oscillate with a sine wave. Use the `model.set_sine_disturbancer` method to add it to the system.
- "hammer attack": given a point in the string, hit it with a hammer. Use the `model.set_hammer_attack` method to add it to the system.
- "noise disturbancer": add noise to the string. Use the `model.set_noise_disturbancer` method to add it to the system.

### Recording
Simulating a omni-directional mic recording the sound of the string at the position `(d0, dist)`.
The distance between the particle i and the mic is $\sqrt{(x_i-d_0)^2 + dist^2}$. So the sound pressure the mic received is vary with different position of the particle.
Anyway, you can see the implementation in `Recorder.step_record`.

