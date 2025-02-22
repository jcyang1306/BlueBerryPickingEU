from copy import copy
 
from ruckig import InputParameter, OutputParameter, Result, Ruckig
import numpy as np
 
if __name__ == '__main__':
    # Create instances: the Ruckig OTG as well as input and output parameters
    kDoFs = 6
    otg = Ruckig(kDoFs, 0.03)  # DoFs, control cycle
    inp = InputParameter(kDoFs)
    out = OutputParameter(kDoFs)
 
    # Set input parameters
    inp.current_position = [0.3495981, -0.0365521,  0.2432336, -3.1324288, -0.0685687, 1.5569506]
    inp.current_velocity = np.zeros(kDoFs)
    inp.current_acceleration = np.zeros(kDoFs)
 
    inp.target_position = [0.2495981, -0.0965521,  0.2732336, -3.1324288, -0.0685687, 1.5569506]
    inp.target_velocity = np.zeros(kDoFs)
    inp.target_acceleration = np.zeros(kDoFs)
 
    inp.max_velocity =  np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.5
    inp.max_acceleration =  np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.1
    inp.max_jerk = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.5
 
    print('\t'.join(['t'] + [str(i) for i in range(otg.degrees_of_freedom)]))
 
    # Generate the trajectory within the control loop
    first_output, out_list = None, []
    res = Result.Working
    while res == Result.Working:
        res = otg.update(inp, out)
 
        print('\t'.join([f'{out.time:0.3f}'] + [f'{p:0.3f}' for p in out.new_position]))
        out_list.append(copy(out))
 
        out.pass_to_input(inp)
 
        if not first_output:
            first_output = copy(out)
 
    print(f'Calculation duration: {first_output.calculation_duration:0.1f} [µs]')
    print(f'Trajectory duration: {first_output.trajectory.duration:0.4f} [s]')
 
    # Plot the trajectory
    # from pathlib import Path
    # from plotter import Plotter
 
    # project_path = Path(__file__).parent.parent.absolute()
    # Plotter.plot_trajectory(project_path / 'examples' / '01_trajectory.pdf', otg, inp, out_list, plot_jerk=False)
