from manim import *
import os
from PIL import Image

class Action(ThreeDScene):
    def construct(self):
        # Configuration dictionary
        config = {
            "image_path": "images/label/ground_truth.png",
            "square_size": 3.0,
            "square_color": GREEN,
            "square_stroke_width": 3,
            "square_fill_opacity": 0,  # Transparent fill
            "image_scale": 0.95,
            "text_content": "Transverse beam distribution",
            "text_font_size": 25,
            "text_color": WHITE,
            "text_position": DOWN * 2,  # Closer to the image
            "final_line": {
                "start": LEFT * 4 + DOWN * 0.2,
                "end": LEFT * 4 + UP * 0.2
            },
            "animation_times": {
                "square_create": 2,
                "wait_after_square": 0.5,
                "image_fade_in": 2,
                "text_write": 1.5,
                "hold_before_transition": 1,
                "transition_duration": 1.5,
                "final_wait": 2
            }
        }
        
        # Create a green transparent square in the center
        square = Square(
            side_length=config["square_size"], 
            color=config["square_color"],
            fill_opacity=config["square_fill_opacity"],
            stroke_width=config["square_stroke_width"]
        )
        square.move_to(ORIGIN)
        
        # Create the image object - simplified approach
        image = ImageMobject(config["image_path"])
        # Scale the image to fit the square
        image.scale_to_fit_width(config["square_size"] * config["image_scale"])
        image.scale_to_fit_height(config["square_size"] * config["image_scale"])
        image.move_to(ORIGIN)
        
        # Create text at the bottom
        text = Text(
            config["text_content"], 
            font_size=config["text_font_size"], 
            color=config["text_color"]
        )
        text.move_to(config["text_position"])
        
        # Create the target line for final transition
        target_line = Line(
            start=config["final_line"]["start"],
            end=config["final_line"]["end"],
            color=config["square_color"],
            stroke_width=config["square_stroke_width"]
        )
        
        # Animation sequence
        # 1. Show the square with typical manim creation animation
        self.play(Create(square), run_time=config["animation_times"]["square_create"])
        
        # 2. Wait a moment
        self.wait(config["animation_times"]["wait_after_square"])
        
        # 3. Gradually show the image fitting into the square
        self.play(FadeIn(image), run_time=config["animation_times"]["image_fade_in"])
        
        # 4. Show the text at the bottom
        self.play(Write(text), run_time=config["animation_times"]["text_write"])
        
        # 5. Hold the current state before transition
        self.wait(config["animation_times"]["hold_before_transition"])
        
        # 6. Synchronized transition: rotate/squeeze both image and square into a line
        # Create a group to keep image and square synchronized
        image_frame_group = Group(image, square)
        
        # Transform both image and square to the target line position and orientation
        target_center = (config["final_line"]["start"] + config["final_line"]["end"]) / 2
        target_height = np.linalg.norm(
            np.array(config["final_line"]["end"]) - 
            np.array(config["final_line"]["start"])
        )
        
        # First, quickly fade out the text
        self.play(FadeOut(text), run_time=0.3)
        
        # Then do the main transformation
        self.play(
            # Combine all transformations in one animation
            image_frame_group.animate
                .rotate(PI/6, axis=UP)  # 3D-like rotation around Y-axis
                .scale([0.05, 1, 1])    # Squeeze horizontally (into center line)
                .scale_to_fit_height(target_height)  # Scale to target line height
                .move_to(target_center),  # Move to target position
            run_time=config["animation_times"]["transition_duration"]
        )
        
        # Create the final line that exactly matches scene1.py ScreenSource
        final_line = Line(
            start=config["final_line"]["start"],
            end=config["final_line"]["end"],
            color=config["square_color"],
            stroke_width=config["square_stroke_width"]
        )
        
        # Smoothly replace the squeezed group with the pure line
        self.play(
            FadeOut(image_frame_group),
            FadeIn(final_line),
            run_time=0.5
        )
        
        # 7. Hold the final frame
        self.wait(config["animation_times"]["final_wait"])