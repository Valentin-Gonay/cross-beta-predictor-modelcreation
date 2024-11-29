### Compute progression time

import time
from typing import Union


def print_loading_bar(
        iteration: int, 
        total: int, 
        prefix: str = '', 
        suffix: str = '', 
        length: int = 50, 
        fill: str = '█'
        ):
    '''Print the progress bar as well as an estimation of the remaining time
    
    :param iteration: The current iteration 
    :type iteration: int 

    :param total: The total number of iteration required to reach 100%
    :type total: int 

    :param prefix: The text to display before the progress bar. If not provided, Defaults to ''
    :type prefix: str 

    :param suffix: The text to display after the loading bar 
    (and before the remaining time estimation). If not provided, Defaults to ''
    :type suffix: str 

    :param length: The size of the progress bar. If not provided, Defaults to 50
    :type length: int 

    :param fill: The character use to fill the progress bar. If not provided, Defaults to '█'
    :type fill: str 
    '''

    percent = f'{(iteration / total) * 100:.1f}'
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    # Calculate time elapsed and estimated remaining time
    elapsed_time = time.time() - print_loading_bar.start_time
    avg_time_per_iteration = elapsed_time / iteration if iteration > 0 else 0
    remaining_iterations = total - iteration
    remaining_time = remaining_iterations * avg_time_per_iteration

    # Format remaining time
    remaining_time_str = convert_sec_to_hh_mm_ss(remaining_time)

    print(f'\r{prefix} [{bar}] {percent}% {suffix} ETA: {remaining_time_str}', end='', flush=True)



def convert_sec_to_hh_mm_ss(sec: Union[int, float]):
    '''Convert a time in seconds into a dd:hh:mm:ss formated time
    
    :param sec: The time in seconds
    :type sec: int | float
    
    :return: The formated time
    :rtype: str
    '''
    
    d = int(sec // 86400)
    h = int((sec % 86400) // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f'{d} day(s) {int(h):02}:{int(m):02}:{s:02}'