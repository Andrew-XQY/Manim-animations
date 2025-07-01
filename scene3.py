from manim import *
import numpy as np
import random
import os
import glob

# === CONFIGURATION ===
config = {
    "network": {
        "neurons_per_layer": [1, 4, 6, 8, 10, 10, 8, 6, 4, 1], # [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]
        "layer_spacing": 1,  # Distance between adjacent layers
        "vertical_offset": -1.0,  # Move network further down to make room for top images
        "neuron_radius": 0.08,
        "neuron_vertical_spacing": [-1.5, 1.5],  # [min, max] vertical range for neurons
        "connection_strength_range": [0.3, 0.7],  # [min, max] opacity
    },
    "images": {
        "top": {
            "y_start": 2.7,  # Starting position above network
            "available_height": 3.0,  # Total vertical space for top images
            "max_image_height": 0.5,  # Maximum height per image (increased)
            "vertical_spacing_factor": 0.6,  # How close images are (reduced spacing)
        },
        "input_output": {
            "image_height": 1,  # Height of input/output images (larger)
            "horizontal_offset": 1,  # Distance from first/last neuron horizontally
        },
        "features_path": "images/features",
    },
    "colors": {
        "neuron": BLUE,
        "connection": WHITE,
        "signal": WHITE,
        "neuron_shine": YELLOW,
        "placeholder_top": RED,
    },
    "animation": {
        "network_creation_time": 2.0,  # Total time for all neurons to appear
        "connection_creation_time": 4.0,  # Total time for all connections to appear
        "dot_propagation_time": 0.9,
        "neuron_shine_time": 0.4,
        "layer_lag_ratio": 0.1,
        "neuron_lag_ratio": 0.05,
        "pause_between_layers": 0.1,  # Pause between layer propagations
        "final_cleanup_time": 0.3,  # Time for final dot cleanup
        "max_dots_per_layer": 8,  # Limit dots to prevent exponential growth
        "dot_radius": 0.04,  # Size of propagation dots
        "dot_opacity": 0.6,  # Transparency of propagation dots
    }
}

class Action(Scene):
    def construct(self):
        # Use configuration
        neurons_per_layer = config["network"]["neurons_per_layer"]
        
        # Configuration for the neural network - automatically centered
        num_layers = len(neurons_per_layer)
        layer_spacing = config["network"]["layer_spacing"]
        total_width = (num_layers - 1) * layer_spacing
        self.layer_positions = np.linspace(-total_width/2, total_width/2, num_layers)  # Auto-centered based on layer count
        
        # Apply vertical offset to move network down
        self.vertical_offset = config["network"]["vertical_offset"]
        
        # Store config for use in other methods
        self.config = config
        
        # Create the neural network structure
        self.create_network(neurons_per_layer)
        
        # Load and prepare feature images
        self.load_feature_images(neurons_per_layer)
        
        # Show the network structure layer by layer
        self.animate_network_creation()
        
        self.wait(1)
        
        # Animate dot propagation with synchronized image features
        self.animate_dot_propagation_with_images()
        
        self.wait(1)
        
        # Fade out everything except the output image for smooth transition
        self.fade_to_output_only()
        
        self.wait(1)

    def create_network(self, neurons_per_layer):
        self.layers = []
        self.connections = []
        self.neurons = []  # Store individual neurons for animation
        
        # Create layers
        for i, (x_pos, num_neurons) in enumerate(zip(self.layer_positions, neurons_per_layer)):
            layer_group = VGroup()
            layer_neurons = []
            
            # Calculate vertical positions for neurons in this layer
            if num_neurons == 1:
                y_positions = [0]
            else:
                y_min, y_max = config["network"]["neuron_vertical_spacing"]
                y_positions = np.linspace(y_min, y_max, num_neurons)
            
            # Create neurons for this layer
            for j, y_pos in enumerate(y_positions):
                neuron = Circle(
                    radius=config["network"]["neuron_radius"],
                    color=config["colors"]["neuron"],
                    fill_opacity=0.3,
                    stroke_width=2
                ).move_to([x_pos, y_pos + self.vertical_offset, 0])  # Apply vertical offset
                
                layer_group.add(neuron)
                layer_neurons.append(neuron)
            
            self.layers.append(layer_group)
            self.neurons.append(layer_neurons)
        
        # Create connections between adjacent layers - fully connected
        for i in range(len(self.neurons) - 1):
            current_layer = self.neurons[i]
            next_layer = self.neurons[i + 1]
            
            # Connect every neuron in current layer to every neuron in next layer
            for neuron1 in current_layer:
                for neuron2 in next_layer:
                    # Random connection strength from config
                    strength_min, strength_max = config["network"]["connection_strength_range"]
                    strength = random.uniform(strength_min, strength_max)
                    
                    connection = Line(
                        neuron1.get_center(),
                        neuron2.get_center(),
                        color=config["colors"]["connection"],
                        stroke_width=1,
                        stroke_opacity=strength
                    )
                    
                    self.connections.append(connection)

    def animate_network_creation(self):
        """Animate the network creation with continuous left-to-right flow"""
        
        # Calculate timing based on config
        total_neuron_time = config["animation"]["network_creation_time"]
        total_connection_time = config["animation"]["connection_creation_time"]
        
        # Collect all neurons in left-to-right order
        all_neurons_ordered = []
        for layer_idx, layer in enumerate(self.layers):
            for neuron_idx, neuron in enumerate(layer):
                all_neurons_ordered.append((layer_idx, neuron_idx, neuron))
        
        # Calculate timing for each neuron
        neuron_interval = total_neuron_time / len(all_neurons_ordered) if len(all_neurons_ordered) > 0 else 0.1
        
        # Phase 1: Animate neurons appearing from left to right
        neuron_animations = []
        for i, (layer_idx, neuron_idx, neuron) in enumerate(all_neurons_ordered):
            delay = i * neuron_interval
            
            # Create the neuron animation with delay
            neuron_animations.append(
                AnimationGroup(
                    Wait(delay),
                    GrowFromCenter(neuron).set_rate_func(smooth)
                )
            )
        
        # Start all neuron animations
        self.play(
            *neuron_animations,
            run_time=total_neuron_time + 0.6  # Extra time for last neuron to complete
        )
        
        # Phase 2: Connections appear from left to right, starting when source neuron is ready
        # Calculate when each neuron becomes available (when it finishes growing)
        neuron_ready_times = {}
        for i, (layer_idx, neuron_idx, neuron) in enumerate(all_neurons_ordered):
            ready_time = i * neuron_interval + 0.6  # Growth time
            neuron_ready_times[(layer_idx, neuron_idx)] = ready_time
        
        # Create connection animations that start when source neuron is ready
        connection_animations = []
        
        for layer_idx in range(len(self.neurons) - 1):
            current_layer = self.neurons[layer_idx]
            next_layer = self.neurons[layer_idx + 1]
            
            # For each neuron in current layer
            for current_neuron_idx, current_neuron in enumerate(current_layer):
                # Find when this neuron is ready
                current_ready_time = neuron_ready_times.get((layer_idx, current_neuron_idx), 0)
                
                # Find connections from this neuron
                neuron_connections = []
                for connection in self.connections:
                    if np.allclose(connection.get_start(), current_neuron.get_center(), atol=0.01):
                        neuron_connections.append(connection)
                
                if neuron_connections:
                    # Calculate connection growth time
                    connection_growth_time = 0.4  # Fixed time for each connection to grow
                    
                    # Create connection animations starting when source is ready
                    for connection in neuron_connections:
                        connection_animations.append(
                            AnimationGroup(
                                Wait(current_ready_time),
                                Create(connection, run_time=connection_growth_time, rate_func=smooth)
                            )
                        )
        
        # Calculate total animation time for connections
        max_connection_time = 0
        if neuron_ready_times:
            max_neuron_ready = max(neuron_ready_times.values())
            max_connection_time = max_neuron_ready + 0.4  # connection growth time
        
        # Play all connection animations
        if connection_animations:
            self.play(
                *connection_animations,
                run_time=max_connection_time + 0.5
            )

    def load_feature_images(self, neurons_per_layer):
        """Load feature images from the features folder"""
        self.feature_images = {}
        self.input_image = None
        self.output_image = None
        self.input_image_object = None
        self.output_image_object = None
        
        features_path = config["images"]["features_path"]
        
        # Load images for each layer (first n layers matching neuron layers)
        for layer_idx in range(min(len(neurons_per_layer), 10)):  # Max 10 folders available
            layer_path = os.path.join(features_path, str(layer_idx))
            if os.path.exists(layer_path):
                # Get all images in this folder
                image_files = glob.glob(os.path.join(layer_path, "*.png"))
                image_files.sort()  # Sort for consistent ordering
                
                if image_files:
                    # Special handling for input (first) and output (last) layers
                    if layer_idx == 0:  # Input layer
                        self.input_image = image_files[0]  # Use first image only
                    elif layer_idx == len(neurons_per_layer) - 1:  # Output layer
                        self.output_image = image_files[0]  # Use first image only
                    else:
                        # For middle layers, load all images for top display
                        self.feature_images[layer_idx] = image_files
        
        # Create input image (left of first neuron)
        if self.input_image and len(self.layer_positions) > 0:
            input_neuron_pos = [self.layer_positions[0], 0 + self.vertical_offset, 0]
            input_x = input_neuron_pos[0] - config["images"]["input_output"]["horizontal_offset"]
            input_y = input_neuron_pos[1]  # Same vertical level as network center
            
            try:
                print(f"Loading input image: {self.input_image}")
                img = ImageMobject(self.input_image)
                img.height = config["images"]["input_output"]["image_height"]
                img.move_to([input_x, input_y, 0])
                img.set_opacity(0)  # Start invisible
                self.input_image_object = img
                self.add(img)
                print(f"Successfully loaded input image: {self.input_image}")
            except Exception as e:
                print(f"Failed to load input image {self.input_image}: {e}")
                # Create placeholder
                placeholder = Rectangle(
                    width=config["images"]["input_output"]["image_height"],
                    height=config["images"]["input_output"]["image_height"],
                    color=config["colors"]["placeholder_top"],
                    fill_opacity=0.5,
                    stroke_opacity=1.0,
                    stroke_color=WHITE
                ).move_to([input_x, input_y, 0])
                
                label = Text("INPUT", font_size=16, color=WHITE)
                label.move_to([input_x, input_y, 0])
                placeholder_group = VGroup(placeholder, label)
                placeholder_group.set_opacity(0)
                self.input_image_object = placeholder_group
                self.add(placeholder_group)
        
        # Create output image (right of last neuron)
        if self.output_image and len(self.layer_positions) > 0:
            output_neuron_pos = [self.layer_positions[-1], 0 + self.vertical_offset, 0]
            output_x = output_neuron_pos[0] + config["images"]["input_output"]["horizontal_offset"]
            output_y = output_neuron_pos[1]  # Same vertical level as network center
            
            try:
                print(f"Loading output image: {self.output_image}")
                img = ImageMobject(self.output_image)
                img.height = config["images"]["input_output"]["image_height"]
                img.move_to([output_x, output_y, 0])
                img.set_opacity(0)  # Start invisible
                self.output_image_object = img
                self.add(img)
                print(f"Successfully loaded output image: {self.output_image}")
            except Exception as e:
                print(f"Failed to load output image {self.output_image}: {e}")
                # Create placeholder
                placeholder = Rectangle(
                    width=config["images"]["input_output"]["image_height"],
                    height=config["images"]["input_output"]["image_height"],
                    color=config["colors"]["placeholder_top"],
                    fill_opacity=0.5,
                    stroke_opacity=1.0,
                    stroke_color=WHITE
                ).move_to([output_x, output_y, 0])
                
                label = Text("OUTPUT", font_size=16, color=WHITE)
                label.move_to([output_x, output_y, 0])
                placeholder_group = VGroup(placeholder, label)
                placeholder_group.set_opacity(0)
                self.output_image_object = placeholder_group
                self.add(placeholder_group)
        
        # Prepare image objects for top display
        self.top_image_groups = {}
        for layer_idx, image_files in self.feature_images.items():
            if layer_idx < len(self.layer_positions):
                x_pos = self.layer_positions[layer_idx]
                num_images = len(image_files)
                
                # Calculate image size and positions using config values
                available_height = config["images"]["top"]["available_height"]
                max_image_height = config["images"]["top"]["max_image_height"]
                image_height = min(available_height / max(num_images, 1), max_image_height)
                
                # Position images vertically above the layer
                y_start = config["images"]["top"]["y_start"]
                if num_images == 1:
                    y_positions = [y_start]
                else:
                    spacing_factor = config["images"]["top"]["vertical_spacing_factor"]
                    effective_height = available_height * spacing_factor
                    y_positions = np.linspace(y_start, y_start - effective_height + image_height, num_images)
                
                image_group = Group()  # Use Group instead of VGroup for ImageMobjects
                for i, (image_file, y_pos) in enumerate(zip(image_files, y_positions)):
                    try:
                        print(f"Loading image: {image_file}")  # Debug print
                        img = ImageMobject(image_file)
                        # Use proper attribute setting instead of deprecated method
                        img.height = image_height
                        img.move_to([x_pos, y_pos, 0])
                        img.set_opacity(0)  # Start invisible
                        image_group.add(img)
                        print(f"Successfully loaded: {image_file}")  # Debug print
                    except Exception as e:
                        print(f"Failed to load image {image_file}: {e}")  # Debug print
                        # If image can't be loaded, create a visible placeholder
                        placeholder = Rectangle(
                            width=image_height,  # Square placeholder
                            height=image_height,
                            color=config["colors"]["placeholder_top"],  # Use config color
                            fill_opacity=0.5,
                            stroke_opacity=1.0,
                            stroke_color=WHITE
                        ).move_to([x_pos, y_pos, 0])
                        
                        # Add a text label to identify it's a placeholder
                        label = Text(f"L{layer_idx}_{i}", font_size=12, color=WHITE)
                        label.move_to([x_pos, y_pos, 0])
                        placeholder_group = VGroup(placeholder, label)
                        placeholder_group.set_opacity(0)
                        image_group.add(placeholder_group)
                
                self.top_image_groups[layer_idx] = image_group
                self.add(image_group)

    def animate_dot_propagation(self):
        """Animate dots propagating through the network from left to right"""
        
        # Build a mapping of connections for efficient lookup
        self.connection_map = {}
        
        for layer_idx in range(len(self.neurons) - 1):
            current_layer = self.neurons[layer_idx]
            next_layer = self.neurons[layer_idx + 1]
            
            for neuron_idx, neuron1 in enumerate(current_layer):
                # Fully connected: each neuron connects to ALL neurons in the next layer
                connected_neurons = list(range(len(next_layer)))
                self.connection_map[(layer_idx, neuron_idx)] = connected_neurons
        
        # Start with one dot at the single input neuron
        active_dots = []
        input_neuron = self.neurons[0][0]  # First layer has only 1 neuron
        
        # Initialize comprehensive dot tracking if not already done
        if not hasattr(self, 'all_dots'):
            self.all_dots = []
        
        dot = Dot(
            radius=config["animation"]["dot_radius"],
            color=config["colors"]["signal"],
            fill_opacity=config["animation"]["dot_opacity"]
        ).move_to(input_neuron.get_center())
        active_dots.append(dot)
        self.all_dots.append(dot)  # Track this dot
        self.add(dot)
        
        # Make input neuron shine
        self.play(
            input_neuron.animate.set_fill(config["colors"]["neuron_shine"], 0.8).set_stroke(config["colors"]["neuron_shine"], width=3),
            run_time=config["animation"]["neuron_shine_time"]
        )
        self.play(
            input_neuron.animate.set_fill(config["colors"]["neuron"], 0.3).set_stroke(config["colors"]["neuron"], width=2),
            run_time=config["animation"]["neuron_shine_time"]
        )
        
        # Propagate through each layer
        for layer_idx in range(len(self.neurons) - 1):
            current_layer = self.neurons[layer_idx]
            next_layer = self.neurons[layer_idx + 1]
            
            # Limit the number of dots to prevent exponential growth
            if len(active_dots) > config["animation"]["max_dots_per_layer"]:
                # Keep only a random subset of dots
                active_dots = random.sample(active_dots, config["animation"]["max_dots_per_layer"])
            
            new_dots = []
            dot_animations = []
            neurons_to_shine = set()
            dots_to_remove = []  # Track dots for explicit removal
            
            # Create a position-to-neuron mapping for efficient lookup
            neuron_positions = {}
            for neuron_idx, neuron in enumerate(current_layer):
                pos_key = (round(neuron.get_center()[0], 2), round(neuron.get_center()[1], 2))
                neuron_positions[pos_key] = neuron_idx
            
            # For each current dot, create new dots following actual connections
            for current_dot in active_dots:
                # Find which neuron this dot is on using efficient lookup
                current_pos = current_dot.get_center()
                pos_key = (round(current_pos[0], 2), round(current_pos[1], 2))
                current_neuron_idx = neuron_positions.get(pos_key)
                
                if current_neuron_idx is not None:
                    # Get connected neuron indices from the connection map
                    connected_indices = self.connection_map.get((layer_idx, current_neuron_idx), [])
                    
                    # Create new dots for each connected neuron
                    for next_neuron_idx in connected_indices:
                        next_neuron = next_layer[next_neuron_idx]
                        neurons_to_shine.add(next_neuron)
                        
                        new_dot = Dot(
                            radius=config["animation"]["dot_radius"],
                            color=config["colors"]["signal"],
                            fill_opacity=config["animation"]["dot_opacity"]
                        ).move_to(current_layer[current_neuron_idx].get_center())
                        
                        new_dots.append(new_dot)
                        self.add(new_dot)
                        
                        # Animate dot moving along connection
                        dot_animations.append(
                            new_dot.animate.move_to(next_neuron.get_center())
                        )
                
                # Mark current dot for immediate removal (no fade animation needed)
                dots_to_remove.append(current_dot)
            
            # Immediately remove old dots when new dots start moving
            for dot in dots_to_remove:
                dot.set_opacity(0)
                dot.set_fill(opacity=0)
                dot.set_stroke(opacity=0)
                self.remove(dot)
            
            # Use comprehensive cleanup BEFORE animation to catch any missed dots
            if hasattr(self, 'cleanup_all_dots'):
                self.cleanup_all_dots(keep_dots=new_dots)
            
            # Animate all dots moving to next layer (old dots already removed)
            if dot_animations:
                # Remove FadeOut animations from dot_animations since we handled cleanup above
                movement_animations = [anim for anim in dot_animations if not isinstance(anim, FadeOut)]
                if movement_animations:
                    self.play(
                        AnimationGroup(*movement_animations),
                        run_time=config["animation"]["dot_propagation_time"]
                    )
            
            # Cleanup is already done before animation, no need to repeat here
            
            # Make next layer neurons shine when dots arrive (only those that received dots)
            if neurons_to_shine:
                shine_animations = []
                for neuron in neurons_to_shine:
                    shine_animations.append(
                        neuron.animate.set_fill(config["colors"]["neuron_shine"], 0.8).set_stroke(config["colors"]["neuron_shine"], width=3)
                    )
                
                self.play(
                    AnimationGroup(*shine_animations),
                    run_time=config["animation"]["neuron_shine_time"]
                )
                
                # Fade neurons back to normal
                fade_animations = []
                for neuron in neurons_to_shine:
                    fade_animations.append(
                        neuron.animate.set_fill(config["colors"]["neuron"], 0.3).set_stroke(config["colors"]["neuron"], width=2)
                    )
                
                self.play(
                    AnimationGroup(*fade_animations),
                    run_time=config["animation"]["neuron_shine_time"]
                )
            
            # Update active dots for next iteration - only keep dots that arrived at neurons
            active_dots = new_dots
            
            self.wait(config["animation"]["pause_between_layers"])
        
        # Final cleanup - remove any remaining dots
        if active_dots:
            self.play(
                *[FadeOut(dot) for dot in active_dots],
                run_time=config["animation"]["final_cleanup_time"]
            )

    def animate_dot_propagation_with_images(self):
        """Animate dots propagating through the network with synchronized image features"""
        
        # Initialize comprehensive dot tracking
        self.all_dots = []  # Track ALL dots ever created for cleanup
        
        # Build a mapping of connections for efficient lookup
        self.connection_map = {}
        
        for layer_idx in range(len(self.neurons) - 1):
            current_layer = self.neurons[layer_idx]
            next_layer = self.neurons[layer_idx + 1]
            
            for neuron_idx, neuron1 in enumerate(current_layer):
                # Fully connected: each neuron connects to ALL neurons in the next layer
                connected_neurons = list(range(len(next_layer)))
                self.connection_map[(layer_idx, neuron_idx)] = connected_neurons
        
        # Start with one dot at the single input neuron
        active_dots = []
        input_neuron = self.neurons[0][0]
        
        dot = Dot(
            radius=config["animation"]["dot_radius"],
            color=config["colors"]["signal"],
            fill_opacity=config["animation"]["dot_opacity"]
        ).move_to(input_neuron.get_center())
        active_dots.append(dot)
        self.all_dots.append(dot)  # Track this dot
        self.add(dot)
        
        # Show first layer images (top only for middle layers, side image for input)
        layer_0_animations = []
        
        # Show input image (on the left)
        if self.input_image_object is not None:
            layer_0_animations.append(self.input_image_object.animate.set_opacity(1))
        
        # Show top images for layer 0 if it's not input/output (but with our config it is, so this won't run)
        if 0 in self.top_image_groups:
            for img in self.top_image_groups[0]:
                layer_0_animations.append(img.animate.set_opacity(1))
        
        # Make input neuron shine and show first images
        self.play(
            input_neuron.animate.set_fill(config["colors"]["neuron_shine"], 0.8).set_stroke(config["colors"]["neuron_shine"], width=3),
            *layer_0_animations,
            run_time=config["animation"]["neuron_shine_time"]
        )
        self.play(
            input_neuron.animate.set_fill(config["colors"]["neuron"], 0.3).set_stroke(config["colors"]["neuron"], width=2),
            run_time=config["animation"]["neuron_shine_time"]
        )
        
        # Propagate through each layer
        for layer_idx in range(len(self.neurons) - 1):
            current_layer = self.neurons[layer_idx]
            next_layer = self.neurons[layer_idx + 1]
            
            # Limit the number of dots to prevent exponential growth
            if len(active_dots) > config["animation"]["max_dots_per_layer"]:
                # Keep only a random subset of dots
                active_dots = random.sample(active_dots, config["animation"]["max_dots_per_layer"])
            
            new_dots = []
            dot_animations = []
            neurons_to_shine = set()
            image_animations = []
            dots_to_remove = []  # Track dots for explicit removal
            
            # Create a position-to-neuron mapping for efficient lookup
            neuron_positions = {}
            for neuron_idx, neuron in enumerate(current_layer):
                pos_key = (round(neuron.get_center()[0], 2), round(neuron.get_center()[1], 2))
                neuron_positions[pos_key] = neuron_idx
            
            # For each current dot, create new dots following actual connections
            for current_dot in active_dots:
                # Find which neuron this dot is on using efficient lookup
                current_pos = current_dot.get_center()
                pos_key = (round(current_pos[0], 2), round(current_pos[1], 2))
                current_neuron_idx = neuron_positions.get(pos_key)
                
                if current_neuron_idx is not None:
                    # Get connected neuron indices from the connection map
                    connected_indices = self.connection_map.get((layer_idx, current_neuron_idx), [])
                    
                    # Create new dots for each connected neuron
                    for next_neuron_idx in connected_indices:
                        next_neuron = next_layer[next_neuron_idx]
                        neurons_to_shine.add(next_neuron)
                        
                        new_dot = Dot(
                            radius=config["animation"]["dot_radius"],
                            color=config["colors"]["signal"],
                            fill_opacity=config["animation"]["dot_opacity"]
                        ).move_to(current_layer[current_neuron_idx].get_center())
                        
                        new_dots.append(new_dot)
                        self.add(new_dot)
                        
                        # Animate dot moving along connection
                        dot_animations.append(
                            new_dot.animate.move_to(next_neuron.get_center())
                        )
                
                # Mark current dot for immediate removal (no fade animation needed)
                dots_to_remove.append(current_dot)
            
            # Show next layer images when dots start moving
            next_layer_idx = layer_idx + 1
            
            # Handle output layer specially (show side image instead of top images)
            if next_layer_idx == len(config["network"]["neurons_per_layer"]) - 1:  # Last layer (output)
                if self.output_image_object is not None:
                    image_animations.append(self.output_image_object.animate.set_opacity(1))
            else:
                # Show top images for middle layers
                if next_layer_idx in self.top_image_groups:
                    for img in self.top_image_groups[next_layer_idx]:
                        image_animations.append(img.animate.set_opacity(1))
            
            # Immediately remove old dots when new dots start moving
            for dot in dots_to_remove:
                dot.set_opacity(0)
                dot.set_fill(opacity=0)
                dot.set_stroke(opacity=0)
                self.remove(dot)
            
            # Use comprehensive cleanup BEFORE animation to catch any missed dots
            if hasattr(self, 'cleanup_all_dots'):
                self.cleanup_all_dots(keep_dots=new_dots)
            
            # Animate all dots moving to next layer with synchronized images (old dots already removed)
            if dot_animations:
                # Remove FadeOut animations from dot_animations since we handled cleanup above
                movement_animations = [anim for anim in dot_animations if not isinstance(anim, FadeOut)]
                all_animations = movement_animations + image_animations
                if all_animations:
                    self.play(
                        AnimationGroup(*all_animations),
                        run_time=config["animation"]["dot_propagation_time"]
                    )
            
            # Cleanup is already done before animation, no need to repeat here
            
            # Make next layer neurons shine when dots arrive
            if neurons_to_shine:
                shine_animations = []
                for neuron in neurons_to_shine:
                    shine_animations.append(
                        neuron.animate.set_fill(config["colors"]["neuron_shine"], 0.8).set_stroke(config["colors"]["neuron_shine"], width=3)
                    )
                
                self.play(
                    AnimationGroup(*shine_animations),
                    run_time=config["animation"]["neuron_shine_time"]
                )
                
                # Fade neurons back to normal
                fade_animations = []
                for neuron in neurons_to_shine:
                    fade_animations.append(
                        neuron.animate.set_fill(config["colors"]["neuron"], 0.3).set_stroke(config["colors"]["neuron"], width=2)
                    )
                
                self.play(
                    AnimationGroup(*fade_animations),
                    run_time=config["animation"]["neuron_shine_time"]
                )
            
            # Update active dots for next iteration
            active_dots = new_dots
            
            self.wait(config["animation"]["pause_between_layers"])
        
        # Final cleanup - remove any remaining dots
        if active_dots:
            self.play(
                *[FadeOut(dot) for dot in active_dots],
                run_time=config["animation"]["final_cleanup_time"]
            )

    def fade_to_output_only(self):
        """Fade out everything except the output image for smooth transition to next scene"""
        
        # Collect all objects to fade out
        fade_out_objects = []
        
        # Add all neurons from all layers
        for layer in self.layers:
            for neuron in layer:
                fade_out_objects.append(neuron)
        
        # Add all connections
        fade_out_objects.extend(self.connections)
        
        # Add input image if it exists
        if self.input_image_object is not None:
            fade_out_objects.append(self.input_image_object)
        
        # Add all top images
        for layer_idx, image_group in self.top_image_groups.items():
            for img in image_group:
                fade_out_objects.append(img)
        
        # Note: We deliberately do NOT add self.output_image_object to fade_out_objects
        # so it remains visible for the transition to scene1
        
        # Fade out everything except the output image
        if fade_out_objects:
            self.play(
                *[FadeOut(obj) for obj in fade_out_objects],
                run_time=1.5,
                rate_func=smooth
            )
        
        print("Faded out all elements except output image - ready for scene transition")
    
    def cleanup_all_dots(self, keep_dots=None):
        """Comprehensive cleanup of all dots to ensure none remain visible"""
        if keep_dots is None:
            keep_dots = []
        
        # Set to track which dots to keep
        keep_set = set(keep_dots)
        
        # Find all Dot objects in the scene
        all_scene_objects = self.mobjects.copy()
        dots_to_clean = []
        
        def find_dots(obj):
            if isinstance(obj, Dot):
                dots_to_clean.append(obj)
            elif hasattr(obj, 'submobjects'):
                for submob in obj.submobjects:
                    find_dots(submob)
        
        for obj in all_scene_objects:
            find_dots(obj)
        
        # Make all dots not in keep_set completely transparent and remove them
        for dot in dots_to_clean:
            if dot not in keep_set:
                # Force dot to be completely invisible
                dot.set_opacity(0)
                dot.set_fill(opacity=0)
                dot.set_stroke(opacity=0)
                # Remove from scene
                self.remove(dot)
        
        # Update tracking list if it exists
        if hasattr(self, 'all_dots'):
            self.all_dots = list(keep_dots)

