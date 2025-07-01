from manim import *
import numpy as np
import os
import glob

# === CONFIGURATION ===
config = {
    "images": {
        "output_path": "images/features/9",  # Last layer images
        "label_path": "images/label",
        "enlarged_size": 3.0,  # Size of enlarged images
        "comparison_spacing": 2.0,  # Distance between output and ground truth images
        "histogram_spacing": 3.5,  # Distance when making room for histograms
    },
    "histogram": {
        "max_height": 1.0,  # Maximum histogram bar height
        "axis_length": 3.0,  # Length of histogram axes
        "bar_width": 0.05,  # Width of histogram bars
        "colors": {
            "ground_truth": BLUE,
            "output": RED,
        }
    },
    "animation": {
        "fade_out_time": 1.5,  # Time to fade out previous scene elements
        "enlarge_time": 2.0,  # Time to enlarge output image
        "ground_truth_appear_time": 1.5,  # Time for ground truth to appear
        "pause_time": 2.0,  # Final pause to view comparison
        "histogram_spacing_time": 1.5,  # Time to move images apart for histograms
        "pixel_projection_time": 4.0,  # Time for pixel projection animation
        "gaussian_fitting_time": 2.5,  # Time for Gaussian curve transformation
    },
    "colors": {
        "placeholder": RED,
    },
    # Import exact config from scene0.py for position calculation
    "scene0_config": {
        "neurons_per_layer": [1, 4, 6, 8, 10, 10, 8, 6, 4, 1],
        "layer_spacing": 1,
        "vertical_offset": -1.0,
        "horizontal_offset": 1,  # input_output horizontal_offset
        "image_height": 1,  # input_output image_height
    }
}

class PositionCalculator:
    """Calculate exact positions matching scene0.py logic"""
    
    @staticmethod
    def calculate_output_position():
        """Calculate the exact position where output image ends in scene0.py"""
        # Replicate scene0.py position calculation logic
        neurons_per_layer = config["scene0_config"]["neurons_per_layer"]
        layer_spacing = config["scene0_config"]["layer_spacing"]
        vertical_offset = config["scene0_config"]["vertical_offset"]
        horizontal_offset = config["scene0_config"]["horizontal_offset"]
        
        # Calculate layer positions (same as scene0.py)
        num_layers = len(neurons_per_layer)
        total_width = (num_layers - 1) * layer_spacing
        layer_positions = np.linspace(-total_width/2, total_width/2, num_layers)
        
        # Calculate output image position (same logic as scene0.py lines 289-291)
        output_neuron_pos = [layer_positions[-1], 0 + vertical_offset, 0]
        output_x = output_neuron_pos[0] + horizontal_offset
        output_y = output_neuron_pos[1]  # Same vertical level as network center
        
        return [output_x, output_y, 0]

class Action(Scene):
    def construct(self):
        # This scene assumes it follows the neural network scene
        # We'll recreate the final state and then transition
        
        # Load output image (from last layer)
        self.output_image_object = self.load_output_image()
        
        # Load ground truth image
        self.ground_truth_image_object = self.load_ground_truth_image()
        
        # Start the transition animation
        self.animate_comparison_transition()

    def load_output_image(self):
        """Load the output image from the last layer folder"""
        output_path = config["images"]["output_path"]
        
        # Calculate exact position from scene0.py
        exact_position = PositionCalculator.calculate_output_position()
        image_height = config["scene0_config"]["image_height"]
        
        if os.path.exists(output_path):
            # Get first image from output folder
            image_files = glob.glob(os.path.join(output_path, "*.png"))
            image_files.sort()
            
            if image_files:
                try:
                    print(f"Loading output image: {image_files[0]}")
                    print(f"Positioning at exact scene0 location: {exact_position}")
                    img = ImageMobject(image_files[0])
                    img.height = image_height  # Exact same height as scene0
                    img.move_to(exact_position)  # Exact same position as scene0
                    self.add(img)
                    print(f"Successfully loaded output image: {image_files[0]}")
                    return img
                except Exception as e:
                    print(f"Failed to load output image {image_files[0]}: {e}")
        
        # Create placeholder if image loading fails
        placeholder = Rectangle(
            width=image_height,
            height=image_height,
            color=config["colors"]["placeholder"],
            fill_opacity=0.5,
            stroke_opacity=1.0,
            stroke_color=WHITE
        ).move_to(exact_position)
        
        label = Text("OUTPUT", font_size=16, color=WHITE)
        label.move_to(exact_position)
        placeholder_group = VGroup(placeholder, label)
        self.add(placeholder_group)
        return placeholder_group

    def load_ground_truth_image(self):
        """Load the ground truth image from the label folder"""
        label_path = config["images"]["label_path"]
        
        if os.path.exists(label_path):
            # Get first image from label folder
            image_files = glob.glob(os.path.join(label_path, "*.png"))
            image_files.sort()
            
            if image_files:
                try:
                    print(f"Loading ground truth image: {image_files[0]}")
                    img = ImageMobject(image_files[0])
                    img.height = config["images"]["enlarged_size"]
                    # Position on left side of screen
                    img.move_to([-config["images"]["comparison_spacing"], 0, 0])
                    img.set_opacity(0)  # Start invisible
                    self.add(img)
                    print(f"Successfully loaded ground truth image: {image_files[0]}")
                    return img
                except Exception as e:
                    print(f"Failed to load ground truth image {image_files[0]}: {e}")
        
        # Create placeholder if image loading fails
        placeholder = Rectangle(
            width=config["images"]["enlarged_size"],
            height=config["images"]["enlarged_size"],
            color=config["colors"]["placeholder"],
            fill_opacity=0.5,
            stroke_opacity=1.0,
            stroke_color=WHITE
        ).move_to([-config["images"]["comparison_spacing"], 0, 0])
        
        label = Text("GROUND\nTRUTH", font_size=24, color=WHITE)
        label.move_to([-config["images"]["comparison_spacing"], 0, 0])
        placeholder_group = VGroup(placeholder, label)
        placeholder_group.set_opacity(0)
        self.add(placeholder_group)
        return placeholder_group

    def animate_comparison_transition(self):
        """Animate the transition from neural network to comparison view"""
        
        # Step 1: Fade out all previous scene elements (simulated)
        # In practice, this would fade out the neural network, but since this is a separate scene,
        # we'll simulate this with a brief pause
        self.wait(0.5)
        
        # Step 2: Enlarge the output image and move it to the right side
        enlarged_position = [config["images"]["comparison_spacing"], 0, 0]
        
        self.play(
            self.output_image_object.animate.scale(
                config["images"]["enlarged_size"] / 1.0  # Scale from 1.0 to enlarged_size
            ).move_to(enlarged_position),
            run_time=config["animation"]["enlarge_time"],
            rate_func=smooth
        )
        
        # Step 3: Show ground truth image on the left
        self.play(
            self.ground_truth_image_object.animate.set_opacity(1),
            run_time=config["animation"]["ground_truth_appear_time"],
            rate_func=smooth
        )
        
        # Step 4: Add labels for clarity
        output_label = Text("Neural Network Output", font_size=20, color=WHITE)
        output_label.next_to(self.output_image_object, DOWN, buff=0.3)
        
        truth_label = Text("Ground Truth", font_size=20, color=WHITE)
        truth_label.next_to(self.ground_truth_image_object, DOWN, buff=0.3)
        
        self.play(
            Write(output_label),
            Write(truth_label),
            run_time=1.0
        )
        
        # Store labels for later use
        self.output_label = output_label
        self.truth_label = truth_label
        
        # Step 5: Final pause to view the comparison
        self.wait(config["animation"]["pause_time"])
        
        # Step 6: Move images apart to make room for histograms (labels move with images)
        self.animate_histogram_analysis()
    
    def animate_histogram_analysis(self):
        """Animate pixel histogram analysis with Gaussian fitting"""
        
        # Move images further apart
        left_position = [-config["images"]["histogram_spacing"], 0, 0]
        right_position = [config["images"]["histogram_spacing"], 0, 0]
        
        self.play(
            self.ground_truth_image_object.animate.move_to(left_position),
            self.output_image_object.animate.move_to(right_position),
            # Move labels with the images but keep same vertical offset
            self.truth_label.animate.move_to([left_position[0], left_position[1] - config["images"]["enlarged_size"]/2 - 0.3, 0]),
            self.output_label.animate.move_to([right_position[0], right_position[1] - config["images"]["enlarged_size"]/2 - 0.3, 0]),
            run_time=config["animation"]["histogram_spacing_time"],
            rate_func=smooth
        )
        
        # Create histogram axes for both images
        self.create_histogram_axes()
        
        # Create and animate histograms forming from axes
        self.create_and_animate_histograms()
        
        # Apply Gaussian fitting
        self.animate_gaussian_fitting()
    
    def create_histogram_axes(self):
        """Create horizontal and vertical histogram axes for both images"""
        
        # Get image positions
        left_pos = self.ground_truth_image_object.get_center()
        right_pos = self.output_image_object.get_center()
        
        # Colors for each image's histograms
        left_color = config["histogram"]["colors"]["ground_truth"]
        right_color = config["histogram"]["colors"]["output"]
        
        # Axis length
        axis_length = config["histogram"]["axis_length"] / 2
        
        # Create axes for left image (ground truth)
        self.left_h_axis = Line(
            [left_pos[0] - axis_length, left_pos[1] + 2, 0],
            [left_pos[0] + axis_length, left_pos[1] + 2, 0],
            color=left_color,
            stroke_width=3
        )
        self.left_v_axis = Line(
            [left_pos[0] + 2, left_pos[1] - axis_length, 0],
            [left_pos[0] + 2, left_pos[1] + axis_length, 0],
            color=left_color,
            stroke_width=3
        )
        
        # Create axes for right image (output)
        self.right_h_axis = Line(
            [right_pos[0] - axis_length, right_pos[1] + 2, 0],
            [right_pos[0] + axis_length, right_pos[1] + 2, 0],
            color=right_color,
            stroke_width=3
        )
        self.right_v_axis = Line(
            [right_pos[0] + 2, right_pos[1] - axis_length, 0],
            [right_pos[0] + 2, right_pos[1] + axis_length, 0],
            color=right_color,
            stroke_width=3
        )
        
        # Draw axes
        self.play(
            Create(self.left_h_axis),
            Create(self.left_v_axis),
            Create(self.right_h_axis),
            Create(self.right_v_axis),
            run_time=1.5
        )
        
        # Store histogram data structures
        self.left_h_bins = []
        self.left_v_bins = []
        self.right_h_bins = []
        self.right_v_bins = []
        
        # Maximum histogram height
        self.max_hist_height = config["histogram"]["max_height"]
    
    def create_and_animate_histograms(self):
        """Create histogram bars and animate them growing from axes"""
        
        # Create histogram data for both images
        left_x_hist, left_y_hist = self.analyze_image_pixels("left")
        right_x_hist, right_y_hist = self.analyze_image_pixels("right")
        
        # Create histogram bars based on actual data
        self.create_histogram_bars(left_x_hist, left_y_hist, 
                                 self.left_h_axis, self.left_v_axis,
                                 config["histogram"]["colors"]["ground_truth"], "left")
        
        self.create_histogram_bars(right_x_hist, right_y_hist,
                                 self.right_h_axis, self.right_v_axis, 
                                 config["histogram"]["colors"]["output"], "right")
        
        # Animate bars growing from axes
        self.animate_bars_growing_from_axes()

    
    def animate_bars_growing_from_axes(self):
        """Animate histogram bars growing out from their respective axes"""
        
        # Collect all bars and their original properties
        all_bars = self.left_h_bins + self.left_v_bins + self.right_h_bins + self.right_v_bins
        
        # Store original positions and sizes
        original_data = []
        for bar in all_bars:
            original_data.append({
                'center': bar.get_center().copy(),
                'width': bar.width,
                'height': bar.height
            })
        
        # Create scaled-down versions at axis positions
        scaled_bars = []
        for i, bar in enumerate(all_bars):
            # Create a copy of the bar that starts very small
            if bar in self.left_h_bins + self.right_h_bins:
                # Vertical bars on horizontal axis - start with no height
                axis_y = self.left_h_axis.get_center()[1] if bar in self.left_h_bins else self.right_h_axis.get_center()[1]
                
                scaled_bar = Rectangle(
                    width=original_data[i]['width'],
                    height=0.01,  # Very small height
                    color=bar.color,
                    fill_opacity=bar.fill_opacity,
                    stroke_opacity=bar.stroke_opacity,
                    stroke_color=bar.stroke_color,
                    stroke_width=bar.stroke_width
                )
                # Position at axis
                scaled_bar.move_to([original_data[i]['center'][0], axis_y, 0])
                
            else:
                # Horizontal bars on vertical axis - start with no width
                axis_x = self.left_v_axis.get_center()[0] if bar in self.left_v_bins else self.right_v_axis.get_center()[0]
                
                scaled_bar = Rectangle(
                    width=0.01,  # Very small width
                    height=original_data[i]['height'],
                    color=bar.color,
                    fill_opacity=bar.fill_opacity,
                    stroke_opacity=bar.stroke_opacity,
                    stroke_color=bar.stroke_color,
                    stroke_width=bar.stroke_width
                )
                # Position at axis
                scaled_bar.move_to([axis_x, original_data[i]['center'][1], 0])
            
            scaled_bars.append(scaled_bar)
            # Add the scaled bar to scene
            self.add(scaled_bar)
        
        # Create growth animations
        growth_animations = []
        for i, (scaled_bar, original) in enumerate(zip(scaled_bars, original_data)):
            if scaled_bar in scaled_bars[:len(self.left_h_bins)] or scaled_bar in scaled_bars[len(self.left_h_bins)+len(self.left_v_bins):len(self.left_h_bins)+len(self.left_v_bins)+len(self.right_h_bins)]:
                # Vertical bars - grow in height
                target_bar = Rectangle(
                    width=original['width'],
                    height=original['height'],
                    color=scaled_bar.color,
                    fill_opacity=scaled_bar.fill_opacity,
                    stroke_opacity=scaled_bar.stroke_opacity,
                    stroke_color=scaled_bar.stroke_color,
                    stroke_width=scaled_bar.stroke_width
                ).move_to(original['center'])
                
                growth_animations.append(
                    Transform(scaled_bar, target_bar, run_time=0.3)
                )
            else:
                # Horizontal bars - grow in width
                target_bar = Rectangle(
                    width=original['width'],
                    height=original['height'],
                    color=scaled_bar.color,
                    fill_opacity=scaled_bar.fill_opacity,
                    stroke_opacity=scaled_bar.stroke_opacity,
                    stroke_color=scaled_bar.stroke_color,
                    stroke_width=scaled_bar.stroke_width
                ).move_to(original['center'])
                
                growth_animations.append(
                    Transform(scaled_bar, target_bar, run_time=0.3)
                )
        
        # Play growth animations with staggered timing
        self.play(
            *growth_animations,
            run_time=2.0,
            rate_func=smooth
        )
        
        # Update the stored bar references to the new bars
        self.left_h_bins = scaled_bars[:len(self.left_h_bins)]
        self.left_v_bins = scaled_bars[len(self.left_h_bins):len(self.left_h_bins)+len(self.left_v_bins)]
        start_idx = len(self.left_h_bins) + len(self.left_v_bins)
        self.right_h_bins = scaled_bars[start_idx:start_idx+len(self.right_h_bins)]
        self.right_v_bins = scaled_bars[start_idx+len(self.right_h_bins):]
    
    def analyze_image_pixels(self, image_type):
        """Analyze actual pixel distribution of the image"""
        try:
            if image_type == "left":
                # Load ground truth image
                label_path = config["images"]["label_path"]
                image_files = glob.glob(os.path.join(label_path, "*.png"))
            else:
                # Load output image
                output_path = config["images"]["output_path"]
                image_files = glob.glob(os.path.join(output_path, "*.png"))
            
            if image_files:
                import cv2
                # Load the first image
                img_path = image_files[0]
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Convert BGR to RGB and get average of channels
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_gray = np.mean(img_rgb, axis=2)  # Average of RGB channels
                    
                    # Get image dimensions
                    height, width = img_gray.shape
                    
                    # Calculate X histogram (sum each column - horizontal projection)
                    x_histogram = np.sum(img_gray, axis=0)  # Sum along rows (axis=0) = column sums
                    
                    # Calculate Y histogram (sum each row - vertical projection)  
                    y_histogram = np.sum(img_gray, axis=1)  # Sum along columns (axis=1) = row sums
                    
                    # Min-max normalization: remap from [min, max] to [0, 1]
                    x_min, x_max = np.min(x_histogram), np.max(x_histogram)
                    y_min, y_max = np.min(y_histogram), np.max(y_histogram)
                    
                    if x_max > x_min:
                        x_histogram = (x_histogram - x_min) / (x_max - x_min)
                    else:
                        x_histogram = np.zeros_like(x_histogram)
                        
                    if y_max > y_min:
                        y_histogram = (y_histogram - y_min) / (y_max - y_min)
                    else:
                        y_histogram = np.zeros_like(y_histogram)
                    
                    print(f"Analyzed {image_type} image: {img_path}")
                    print(f"Image dimensions: {width}x{height}")
                    print(f"X histogram length: {len(x_histogram)} (should be {width})")
                    print(f"Y histogram length: {len(y_histogram)} (should be {height})")
                    print(f"X histogram range: [{np.min(x_histogram):.3f}, {np.max(x_histogram):.3f}]")
                    print(f"Y histogram range: [{np.min(y_histogram):.3f}, {np.max(y_histogram):.3f}]")
                    
                    return x_histogram, y_histogram
                    
        except ImportError:
            print("OpenCV not available, using simulated data")
        except Exception as e:
            print(f"Error loading image: {e}, using simulated data")
        
        # Fallback to simulated data if image loading fails
        print(f"Using simulated data for {image_type} image")
        return self.create_simulated_histograms(image_type)
    
    def create_simulated_histograms(self, image_type):
        """Create simulated histogram data when real images aren't available"""
        # Simulate typical image dimensions
        width, height = 256, 256
        
        if image_type == "left":
            # Ground truth - simulate a more centered distribution
            x_hist = np.exp(-0.5 * ((np.arange(width) - width*0.5) / (width*0.15))**2)
            y_hist = np.exp(-0.5 * ((np.arange(height) - height*0.5) / (height*0.15))**2)
        else:
            # Output - simulate a slightly different distribution
            x_hist = np.exp(-0.5 * ((np.arange(width) - width*0.6) / (width*0.2))**2)
            y_hist = np.exp(-0.5 * ((np.arange(height) - height*0.4) / (height*0.18))**2)
        
        # Add some noise to make it more realistic
        x_hist += np.random.normal(0, 0.1, width)
        y_hist += np.random.normal(0, 0.1, height)
        
        # Ensure non-negative 
        x_hist = np.clip(x_hist, 0, None)
        y_hist = np.clip(y_hist, 0, None)
        
        # Min-max normalization: remap from [min, max] to [0, 1]
        x_min, x_max = np.min(x_hist), np.max(x_hist)
        y_min, y_max = np.min(y_hist), np.max(y_hist)
        
        if x_max > x_min:
            x_hist = (x_hist - x_min) / (x_max - x_min)
        else:
            x_hist = np.zeros_like(x_hist)
            
        if y_max > y_min:
            y_hist = (y_hist - y_min) / (y_max - y_min)
        else:
            y_hist = np.zeros_like(y_hist)
            
        return x_hist, y_hist
    
    def create_histogram_bars(self, x_histogram, y_histogram, h_axis, v_axis, color, side):
        """Create histogram bars with proper binning and correct visual mapping"""
        
        # Configuration for histogram binning
        num_bins = 64  # Target number of bins for visualization
        axis_length = config["histogram"]["axis_length"]
        max_height = config["histogram"]["max_height"]
        
        # Calculate bin size for X histogram
        original_x_length = len(x_histogram)
        x_bin_size = max(1, original_x_length // num_bins)
        x_num_bins = original_x_length // x_bin_size
        
        # Calculate bin size for Y histogram  
        original_y_length = len(y_histogram)
        y_bin_size = max(1, original_y_length // num_bins)
        y_num_bins = original_y_length // y_bin_size
        
        print(f"Binning {side} image: X {original_x_length}->{x_num_bins} bins (size {x_bin_size}), Y {original_y_length}->{y_num_bins} bins (size {y_bin_size})")
        
        # Bin the X histogram data
        x_binned = []
        for i in range(x_num_bins):
            start_idx = i * x_bin_size
            end_idx = min((i + 1) * x_bin_size, original_x_length)
            bin_value = np.mean(x_histogram[start_idx:end_idx])
            x_binned.append(bin_value)
        x_binned = np.array(x_binned)
        
        # Bin the Y histogram data
        y_binned = []
        for i in range(y_num_bins):
            start_idx = i * y_bin_size
            end_idx = min((i + 1) * y_bin_size, original_y_length)
            bin_value = np.mean(y_histogram[start_idx:end_idx])
            y_binned.append(bin_value)
        y_binned = np.array(y_binned)
        
        # Scale to display height
        x_binned = x_binned * max_height
        y_binned = y_binned * max_height
        
        # Calculate bar dimensions - width should match axis perfectly
        x_bar_width = axis_length / len(x_binned)
        y_bar_width = axis_length / len(y_binned)
        
        h_bars = []
        v_bars = []
        
        # Create X histogram bars (VERTICAL bars on horizontal axis)
        h_axis_start = h_axis.get_start()
        for i, height in enumerate(x_binned):
            # Use actual height, minimum height only for very small values
            actual_height = max(height, 0.005)  # Very small minimum
            
            x_pos = h_axis_start[0] + (i + 0.5) * x_bar_width
            y_pos = h_axis_start[1]
            
            bar = Rectangle(
                width=x_bar_width * 0.95,  # Use almost full width to minimize gaps
                height=actual_height,
                color=color,
                fill_opacity=0.8,
                stroke_opacity=0.2,
                stroke_color=color,
                stroke_width=0.5
            )
            bar.move_to([x_pos, y_pos + actual_height/2, 0])
            h_bars.append(bar)
        
        # Create Y histogram bars (HORIZONTAL bars on vertical axis)
        # FLIP the Y data so bottom of image = bottom of axis
        y_binned_flipped = np.flip(y_binned)
        v_axis_start = v_axis.get_start()
        for i, height in enumerate(y_binned_flipped):
            actual_height = max(height, 0.005)  # Very small minimum
            
            x_pos = v_axis_start[0]
            y_pos = v_axis_start[1] + (i + 0.5) * y_bar_width
            
            bar = Rectangle(
                width=actual_height,
                height=y_bar_width * 0.95,  # Use almost full height to minimize gaps
                color=color,
                fill_opacity=0.8,
                stroke_opacity=0.2,
                stroke_color=color,
                stroke_width=0.5
            )
            bar.move_to([x_pos + actual_height/2, y_pos, 0])
            v_bars.append(bar)
        
        # Store everything for later use
        if side == "left":
            self.left_h_bins = h_bars
            self.left_v_bins = v_bars
            self.left_x_data = x_binned / max_height  # Normalized for Gaussian fitting
            self.left_y_data = y_binned_flipped / max_height  # Flipped and normalized
        else:
            self.right_h_bins = h_bars
            self.right_v_bins = v_bars
            self.right_x_data = x_binned / max_height  # Normalized for Gaussian fitting
            self.right_y_data = y_binned_flipped / max_height  # Flipped and normalized
        
        print(f"Created {len(h_bars)} X bars and {len(v_bars)} Y bars for {side} image")
    
    def animate_gaussian_fitting(self):
        """Apply Gaussian fitting to transform histograms into smooth curves based on actual data"""
        
        # Create Gaussian curves based on actual histogram data
        left_x_gaussian = self.create_gaussian_from_histogram(
            self.left_x_data, self.left_h_axis, 
            config["histogram"]["colors"]["ground_truth"], vertical=False
        )
        left_y_gaussian = self.create_gaussian_from_histogram(
            self.left_y_data, self.left_v_axis,
            config["histogram"]["colors"]["ground_truth"], vertical=True
        )
        
        right_x_gaussian = self.create_gaussian_from_histogram(
            self.right_x_data, self.right_h_axis,
            config["histogram"]["colors"]["output"], vertical=False
        )
        right_y_gaussian = self.create_gaussian_from_histogram(
            self.right_y_data, self.right_v_axis,
            config["histogram"]["colors"]["output"], vertical=True
        )
        
        # Transform histograms to Gaussian curves
        self.play(
            # Fade out histogram bars and replace with curves
            *[FadeOut(bar) for bar in self.left_h_bins],
            *[FadeOut(bar) for bar in self.left_v_bins],
            *[FadeOut(bar) for bar in self.right_h_bins],
            *[FadeOut(bar) for bar in self.right_v_bins],
            Create(left_x_gaussian),
            Create(left_y_gaussian),
            Create(right_x_gaussian),
            Create(right_y_gaussian),
            run_time=config["animation"]["gaussian_fitting_time"],
            rate_func=smooth
        )
        
        # Final pause to view the Gaussian fits
        self.wait(2.0)
        
        # Store the Gaussian curves for final comparison animation
        self.left_x_gaussian = left_x_gaussian
        self.left_y_gaussian = left_y_gaussian
        self.right_x_gaussian = right_x_gaussian
        self.right_y_gaussian = right_y_gaussian
        
        # Final comparison animation
        self.animate_final_gaussian_comparison()
    
    def create_gaussian_from_histogram(self, histogram_data, axis, color, vertical=False):
        """Create a Gaussian curve fitted to actual histogram data using proper 1D Gaussian fitting"""
        
        # Perform actual Gaussian fitting on the histogram data
        try:
            from scipy.optimize import curve_fit
            
            # Define Gaussian function
            def gaussian(x, amplitude, mean, std):
                return amplitude * np.exp(-0.5 * ((x - mean) / std) ** 2)
            
            # Create x values for the histogram data
            x_data = np.arange(len(histogram_data))
            y_data = histogram_data
            
            # Initial parameter guesses
            amplitude_guess = np.max(y_data)
            mean_guess = np.sum(x_data * y_data) / np.sum(y_data) if np.sum(y_data) > 0 else len(x_data) / 2
            std_guess = len(x_data) / 6  # Reasonable guess
            
            # Fit Gaussian
            try:
                popt, _ = curve_fit(gaussian, x_data, y_data, 
                                  p0=[amplitude_guess, mean_guess, std_guess],
                                  maxfev=1000)
                amplitude, mean, std = popt
                print(f"Gaussian fit: amplitude={amplitude:.3f}, mean={mean:.3f}, std={std:.3f}")
            except:
                # Fallback to initial guesses if fitting fails
                amplitude, mean, std = amplitude_guess, mean_guess, std_guess
                print("Gaussian fitting failed, using estimates")
                
        except ImportError:
            print("SciPy not available, using statistical estimates for Gaussian")
            # Fallback: calculate mean and std from histogram
            x_data = np.arange(len(histogram_data))
            y_data = histogram_data
            
            total_weight = np.sum(y_data)
            if total_weight > 0:
                mean = np.sum(x_data * y_data) / total_weight
                variance = np.sum(((x_data - mean) ** 2) * y_data) / total_weight
                std = np.sqrt(variance)
                amplitude = np.max(y_data)
            else:
                mean = len(histogram_data) / 2
                std = len(histogram_data) / 6
                amplitude = 1.0
        
        # Get axis properties
        axis_start = axis.get_start()
        axis_end = axis.get_end()
        axis_length = np.linalg.norm(np.array(axis_end) - np.array(axis_start))
        
        # Normalize parameters to axis coordinates
        mean_normalized = mean / len(histogram_data)
        std_normalized = std / len(histogram_data)
        
        if vertical:
            # Vertical Gaussian (along y-axis) - grows horizontally
            def gaussian_func(t):
                # t goes from 0 to 1 along the axis
                y_val = axis_start[1] + t * (axis_end[1] - axis_start[1])
                
                # Calculate Gaussian value
                gauss_val = amplitude * np.exp(-0.5 * ((t - mean_normalized) / std_normalized) ** 2)
                # Scale to display height
                gauss_val = gauss_val / amplitude * config["histogram"]["max_height"]
                
                return [axis_start[0] + gauss_val, y_val, 0]
            
        else:
            # Horizontal Gaussian (along x-axis) - grows vertically
            def gaussian_func(t):
                # t goes from 0 to 1 along the axis
                x_val = axis_start[0] + t * (axis_end[0] - axis_start[0])
                
                # Calculate Gaussian value
                gauss_val = amplitude * np.exp(-0.5 * ((t - mean_normalized) / std_normalized) ** 2)
                # Scale to display height
                gauss_val = gauss_val / amplitude * config["histogram"]["max_height"]
                
                return [x_val, axis_start[1] + gauss_val, 0]
        
        # Create the curve
        curve = ParametricFunction(
            gaussian_func,
            t_range=[0, 1],
            color=color,
            stroke_width=4
        )
        
        return curve

    def animate_final_gaussian_comparison(self):
        """Final animation: fade out images and reorganize Gaussians for side-by-side comparison"""
        
        # Step 1: Fade out the images and histogram axes
        self.play(
            FadeOut(self.ground_truth_image_object),
            FadeOut(self.output_image_object),
            FadeOut(self.truth_label),
            FadeOut(self.output_label),
            FadeOut(self.left_h_axis),
            FadeOut(self.left_v_axis),
            FadeOut(self.right_h_axis),
            FadeOut(self.right_v_axis),
            run_time=1.5,
            rate_func=smooth
        )
        
        # Step 2: Define target positions and scales for Gaussian comparison
        # Left side: horizontal Gaussians (ground truth and output at same vertical level)
        left_comparison_x = -3.0  # Left side of screen
        comparison_y = 0.0  # Same vertical position for both curves
        
        # Right side: vertical Gaussians (rotated 90° and at same vertical level)
        right_comparison_x = 3.0  # Right side of screen
        
        # Enlargement scale
        enlarge_scale = 1.5
        
        # Step 3: Create target positions for horizontal Gaussians (left side, same vertical level)
        left_h_target_pos = [left_comparison_x, comparison_y, 0]  # Ground truth horizontal
        right_h_target_pos = [left_comparison_x, comparison_y, 0]  # Output horizontal
        
        # Step 4: Create target positions for vertical Gaussians (right side, rotated, same vertical level)
        left_v_target_pos = [right_comparison_x, comparison_y, 0]  # Ground truth vertical (rotated)
        right_v_target_pos = [right_comparison_x, comparison_y, 0]  # Output vertical (rotated)
        
        # Step 5: Animate transition of Gaussians to their final comparison positions
        self.play(
            # Move and scale horizontal Gaussians to same position
            self.left_x_gaussian.animate.scale(enlarge_scale).move_to(left_h_target_pos),
            self.right_x_gaussian.animate.scale(enlarge_scale).move_to(right_h_target_pos),
            
            # Move, rotate (90°), and scale vertical Gaussians to same position
            self.left_y_gaussian.animate.scale(enlarge_scale).rotate(PI/2).move_to(left_v_target_pos),
            self.right_y_gaussian.animate.scale(enlarge_scale).rotate(PI/2).move_to(right_v_target_pos),
            
            run_time=3.0,
            rate_func=smooth
        )
        
        # Step 6: Add section titles only
        left_title = Text("Horizontal Beam Profile", font_size=24, color=WHITE)
        left_title.next_to([left_comparison_x, 0, 0], UP, buff=2.5)
        
        right_title = Text("Vertical Beam Profile", font_size=24, color=WHITE)
        right_title.next_to([right_comparison_x, 0, 0], UP, buff=2.5)
        
        # Animate titles appearing
        self.play(
            Write(left_title),
            Write(right_title),
            run_time=2.0
        )
        
        # Step 7: Final pause to view the comparison
        self.wait(3.0)


# If you want to run this scene independently for testing
if __name__ == "__main__":
    # You can test this scene by running: python -m manim scene1.py ComparisonScene -pql
    pass
