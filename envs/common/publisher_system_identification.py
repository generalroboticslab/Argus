import numpy as np
from publisher import DataPublisher
from tqdm import trange
import time

def sweep(t, a, b, freq_start, freq_end, dt=1/200):
    """
    Generates a smooth, continuous sinusoidal sweep from a start to an end frequency
    and back again.

    The key to avoiding discontinuity is to generate a single, continuous phase
    vector by integrating the complete frequency profile (forward and backward).
    """
    # --- 1. Create the time vector for a one-way sweep ---
    x = np.arange(0, t, dt)

    # --- 2. Create the frequency profile for the full sweep ---
    # Frequency profile for the forward sweep (e.g., 0.8 Hz to 1.5 Hz)
    freq_forward = np.linspace(freq_start, freq_end, num=len(x))

    # The full frequency profile is the forward sweep followed by the backward sweep.
    # We simply append the reversed forward sweep for the backward motion.
    full_freq_profile = np.concatenate((freq_forward, freq_forward[::-1]))

    # --- 3. Calculate a single, continuous phase vector ---
    # Integrate the entire frequency profile to get a smooth phase progression.
    # Multiplying by 2*pi converts frequency (Hz) to angular frequency (rad/s).
    # `np.cumsum(...) * dt` is the numerical integration (integral of frequency is phase).
    continuous_phase = 2 * np.pi * np.cumsum(full_freq_profile) * dt

    # --- 4. Generate the final trajectory ---
    # Define the amplitude and vertical offset for the cosine wave.
    offset = (a + b) / 2
    amplitude = (b - a) / 2
    
    # Generate the cosine wave using the continuous phase.
    # The `+ np.pi` offset ensures the wave starts at its lowest point (`a`).
    y = offset + amplitude * np.cos(continuous_phase + np.pi)
    
    return y

if __name__ == "__main__":
    
    # Sample data to publish
    time_since_start = time.time()

    pi = np.pi

    dt = 1/100

    y = sweep(t=5, a=-0.15, b=0.15, freq_start=0.8, freq_end=1.5, dt=dt)


    def test_data(i,leg_id,command):
        """example test data"""
        return {
            "step": i,
            "leg_id": leg_id,
            # "dof_pos_target": command,
            "action": command,
        }

    # Create a publisher instance
    publisher = DataPublisher(
        # target_url="udp://10.197.149.208:9872", 
        target_url="udp://localhost:9871", 
        encoding="msgpack",broadcast=False,thread=True)
    publisher_2 = DataPublisher( # to jetson
        # target_url="udp://192.168.55.1:9871",
        target_url="udp://152.3.173.234:9871",
        encoding="msgpack",broadcast=False,thread=True)

    # selected_legs = [15,16,17,18,19]  # Example leg IDs

    selected_legs = [6, 10, 11, 15, 16] # Example leg IDs

    # while True:
    for i in trange(len(y)*100):
        publisher_2.publish(test_data(i,selected_legs,y[i%len(y)]))
        time.sleep(dt)  # add a small delay
        publisher.publish(test_data(i,selected_legs,y[i%len(y)]))



    print("Publisher and Receiver have stopped.")