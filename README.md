# Dissertation
Repository for my dissertation project. The project regards the use of a wheg robot for autonomous navigation over terrain.

This dissertation studies autonomous robotic navigation through localized decisions based on previous experience. Nearly all organisms are limited at some point when it comes to movement. An octopus can fit through an aperture that is bigger than its beak. A person can climb over rocks, but only if it has grip points for hands and feet. These environmental barriers define conditions to an organism’s ability to traverse across it. When a path looks too arduous, we will often choose a more accessible route if possible. 

These constraints comply with robotics also. Consider a Mars Rover over 395 million km away from the nearest person to get it unstuck over an obstacle. In addition, it takes 181 seconds for a signal to get from earth to Mars; thus, real-time control over changes in an environment will have taken effect by the time the signal arrives. A robot to self-preserve, like an organism, will need to know its limitations and not attempt tasks of movement outside these constraints. 

When deciding on routes, there may be the scenario that all paths are within the constraints. Some are more simpler to navigate through than others. The agent will need to pick the option which requires the “least” effort. The best-case scenario is defined by the level of complexity of movement necessary to overcome the obstacle. 

We will build an agent to perform this task by using a panoramic image, which forms a prediction on the best route to take via a clock-face prediction method. This chassis will have a back actuator, stabilizer, and neck actuator. Once attempting the terrain, the robot will need to use its chassis features to help it navigate over the landscape. 
