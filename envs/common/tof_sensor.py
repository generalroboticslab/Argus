import torch
def depth_to_tof(depth):
    """Converts depth (meters) to Time-of-Flight units 
    !! User need to convert it to uint8 to match the physical TOF sensor output (0-255)!!
    """
    return (161.276162 * torch.clip(depth, 0, 2.5).sqrt()) #.to(torch.uint8)
    # return 161.276162 * torch.clip(depth, 0, 2.5).sqrt().to(torch.float)


def tof_to_depth(tof):
    """Converts Time-of-Flight units back to depth (meters)"""
    tof_float = tof.astype(float) if hasattr(tof, 'astype') else float(tof)
    depth = (tof_float / 161.276162) ** 2
    return depth

def make_noisy_depth(
        depth,
        max_error=0.025,
        max_outlier_probs=0.01,
        min_outlier_probs=0.005,
        max_depth=2.5
        ):
    """
    Adds Gaussian noise and random outliers to depth measurements
    Returns noisy depth [m].
    """
    probs = torch.empty_like(depth).uniform_()
    depth_with_noise = depth * (torch.empty_like(depth).normal_(1, max_error)) \
         -max_depth*(probs>(1-min_outlier_probs)) \
        + max_depth*(probs<max_outlier_probs)
    return depth_with_noise

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    # evluate the function
    # --- Simulation Parameters ---
    H, W = 25, 25         # Height and Width of the depth sensor
    num_frames = 40      # Number of frames in the animation
    triangle_depth = 0.2  # Using fixed depth from the provided script
    max_depth_sim = 3   # Background depth / max depth for noise/clipping
    fps = 10              # Frames per second for the animation

    # --- Triangle Definition and Movement ---
    # Using vertices & velocity from the provided script
    initial_vertices = torch.tensor([[0, 0], [0, 15], [7, 12]], dtype=torch.float32)
    velocity = torch.tensor([0.1, 0.2], dtype=torch.float32) # Moves slightly down and right

    # --- Helper function for SLOW triangle check (as provided) ---
    # (Consider vectorizing this section if performance is an issue)
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    def is_inside_triangle(pt, v1, v2, v3):
        d1 = sign(pt, v1, v2)
        d2 = sign(pt, v2, v3)
        d3 = sign(pt, v3, v1)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)
    # --- End of slow check function ---

    # --- Data Storage ---
    tof_frames_clean = [] # <-- ADDED: Store clean frames
    tof_frames_noisy = [] # Store noisy frames

    print("Generating frames...")

    # --- Frame Generation Loop ---
    for frame_idx in range(num_frames):
        # Calculate current triangle vertices
        current_vertices = initial_vertices + frame_idx * velocity

        # Create the clean depth map for this frame
        depth_map_clean = torch.full((H, W), max_depth_sim, dtype=torch.float32)

        # Determine which pixels are inside the current triangle (SLOW VERSION)
        v1, v2, v3 = current_vertices[0], current_vertices[1], current_vertices[2]
        inside_mask = torch.zeros((H, W), dtype=torch.bool)
        for r in range(H):
            for c in range(W):
                pt = torch.tensor([r, c], dtype=torch.float32)
                if is_inside_triangle(pt, v1, v2, v3):
                    inside_mask[r, c] = True

        # Assign triangle depth to pixels inside the triangle
        depth_map_clean[inside_mask] = triangle_depth

        # --- Calculate and Store Both Clean and Noisy ToF ---
        tof_map_clean = depth_to_tof(depth_map_clean) # <-- Calculate clean ToF
        tof_frames_clean.append(tof_map_clean)        # <-- Store clean ToF

        depth_map_noisy = make_noisy_depth(depth_map_clean, max_depth=max_depth_sim)
        tof_map_noisy = depth_to_tof(depth_map_noisy)
        tof_frames_noisy.append(tof_map_noisy)
        # --- End ToF Calculation ---

        if (frame_idx + 1) % 10 == 0:
            print(f"Generated frame {frame_idx + 1}/{num_frames}")

    print("Frames generated. Creating animation...")

    # --- Animation Setup (Side-by-Side) ---
    fig, axs = plt.subplots(1, 2, figsize=(10, 5)) # <-- Create 1 row, 2 columns of subplots

    # Use vmin/vmax based on expected ToF range
    tof_min = depth_to_tof(torch.tensor(0.0)).item()
    tof_max = depth_to_tof(torch.tensor(max_depth_sim)).item()

    # Initialize the first plot (Clean ToF)
    im_clean = axs[0].imshow(tof_frames_clean[0].numpy(), cmap='jet', vmin=tof_min, vmax=tof_max)
    axs[0].set_title("Clean ToF (Frame 0)")
    fig.colorbar(im_clean, ax=axs[0], shrink=0.8) # Add colorbar to the first axes

    # Initialize the second plot (Noisy ToF)
    im_noisy = axs[1].imshow(tof_frames_noisy[0].numpy(), cmap='jet', vmin=tof_min, vmax=tof_max)
    axs[1].set_title("Noisy ToF (Frame 0)")
    fig.colorbar(im_noisy, ax=axs[1], shrink=0.8) # Add colorbar to the second axes

    plt.tight_layout() # Adjust layout to prevent overlap

    # --- Animation update function (Updates Both Plots) ---
    def update(frame):
        # Update data for both images
        im_clean.set_data(tof_frames_clean[frame].numpy())
        im_noisy.set_data(tof_frames_noisy[frame].numpy())

        # Update titles for both axes
        axs[0].set_title(f"Clean ToF (Frame {frame})")
        axs[1].set_title(f"Noisy ToF (Frame {frame})")

        # Return both image objects for blitting
        return im_clean, im_noisy,

    # --- Create and Save Animation ---
    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                interval=1000/fps, blit=True)

    # # Save the animation (make sure Pillow is installed: pip install Pillow)
    # gif_filename = 'moving_triangle_tof_side_by_side.gif' # New filename
    # print(f"Saving animation as {gif_filename}...")
    # try:
    #     ani.save(gif_filename, writer='pillow', fps=fps)
    #     print("Animation saved.")
    # except Exception as e:
    #     print(f"Error saving animation: {e}")
    #     print("Make sure Pillow is installed ('pip install Pillow').")

    # --- Show the animation ---
    plt.show()
    print("Done.")