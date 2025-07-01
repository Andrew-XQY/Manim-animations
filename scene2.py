from manim import *

class Action(ThreeDScene):
    def construct(self):
        # Configuration dictionary
        config = {
            # Initial receiver line (exactly matching scene1.py final state)
            "receiver_line": {
                "start": RIGHT * 4 + DOWN * 0.45,  # Exact position from scene1
                "end": RIGHT * 4 + UP * 0.45,      # Exact position from scene1
                "color": YELLOW,
                "stroke_width": 4
            },
            # Rectangle configuration (same height as receiver, zero width initially)
            "rectangle": {
                "height": 0.9,  # Same as receiver line height
                "initial_width": 0.001,  # Almost zero width (appears as line)
                "final_width": 3.0,      # Final square width
                "color": YELLOW,
                "stroke_width": 4,
                "fill_opacity": 0,  # Transparent
                "initial_position": RIGHT * 4,  # Same as receiver position
                "final_position": ORIGIN         # Center of screen
            },
            # Image configuration
            "image": {
                "path": "images/features/0/input_layer_ch0.png",
                "scale": 0.95  # Slightly smaller than the rectangle
            },
            # Animation timing
            "animation_times": {
                "initial_hold": 1.0,
                "line_to_rectangle": 0.5,
                "rectangle_transformation": 1.5,
                "final_hold": 2.0,
                "text_appear": 1.5,
                "shift_left": 2.0,
                "right_elements_appear": 2.0,
                "arrow_appear": 1.0,
                "question_appear": 0.5,
                "final_extended_hold": 3.0,
                "fade_out_all": 2.0
            },
            # Text for the left side (under the square)
            "left_text": {
                "line1": "Multimode fiber",
                "line2": "speckle pattern",
                "font_size": 22,
                "color": WHITE,
                "spacing": 0.4,  # Vertical spacing between lines
                "font": "sans-serif"  # Use a more reliable font
            },
            # Right side elements
            "right_square": {
                "color": BLUE,  # Color from scene1.py wall
                "stroke_width": 4,
                "fill_opacity": 0
            },
            "right_image": {
                "path": "images/label/ground_truth.png",
                "scale": 0.95
            },
            "right_text": {
                "content": "Original distribution",
                "font_size": 22,
                "color": WHITE,
                "font": "sans-serif"  # Use a more reliable font
            },
            # Arrow and question mark
            "arrow": {
                "start": LEFT * 1,
                "end": RIGHT * 1,
                "color": WHITE,
                "stroke_width": 3
            },
            "question_mark": {
                "content": "?",
                "font_size": 48,
                "color": WHITE,
                "font": "sans-serif"  # Use a more reliable font
            },
            # Final positions after shift
            "left_position": LEFT * 3,
            "right_position": RIGHT * 3
        }
        
        # Step 1: Create initial receiver line (exactly matching scene1.py final state)
        receiver_line = Line(
            start=config["receiver_line"]["start"],
            end=config["receiver_line"]["end"],
            color=config["receiver_line"]["color"],
            stroke_width=config["receiver_line"]["stroke_width"]
        )
        
        # Add receiver line to scene
        self.add(receiver_line)
        
        # Hold initial state
        self.wait(config["animation_times"]["initial_hold"])
        
        # Step 2: Create rectangle with same height as receiver but zero width
        initial_rectangle = Rectangle(
            width=config["rectangle"]["initial_width"],
            height=config["rectangle"]["height"],
            color=config["rectangle"]["color"],
            stroke_width=config["rectangle"]["stroke_width"],
            fill_opacity=config["rectangle"]["fill_opacity"]
        )
        initial_rectangle.move_to(config["rectangle"]["initial_position"])
        
        # Create initial image (very thin, same position as rectangle)
        initial_image = ImageMobject(config["image"]["path"])
        # Scale to fit the height first, then squeeze the width
        initial_image.scale_to_fit_height(config["rectangle"]["height"] * config["image"]["scale"])
        initial_image.scale([0.001, 1, 1])  # Squeeze horizontally to match rectangle width
        initial_image.move_to(config["rectangle"]["initial_position"])
        
        # Replace line with rectangle and image (should look identical since width is almost zero)
        self.play(
            FadeOut(receiver_line),
            FadeIn(initial_rectangle),
            FadeIn(initial_image),
            run_time=config["animation_times"]["line_to_rectangle"]
        )
        
        # Step 3: Create final square (larger, at center)
        final_square = Rectangle(
            width=config["rectangle"]["final_width"],
            height=config["rectangle"]["final_width"],  # Square: width = height
            color=config["rectangle"]["color"],
            stroke_width=config["rectangle"]["stroke_width"],
            fill_opacity=config["rectangle"]["fill_opacity"]
        )
        final_square.move_to(config["rectangle"]["final_position"])
        
        # Create final image (fitting the final square)
        final_image = ImageMobject(config["image"]["path"])
        final_image.scale_to_fit_width(config["rectangle"]["final_width"] * config["image"]["scale"])
        final_image.scale_to_fit_height(config["rectangle"]["final_width"] * config["image"]["scale"])
        final_image.move_to(config["rectangle"]["final_position"])
        
        # Step 4: Simultaneous transformation for both rectangle and image:
        # - Move to center
        # - Unsqueeze (increase width gradually) 
        # - Scale up to final square size
        self.play(
            # Rectangle transformation - use Transform for consistent scaling
            Transform(initial_rectangle, final_square),
            # Image transformation with explicit unsqueeze - use Transform for consistent scaling
            Transform(initial_image, final_image),
            run_time=config["animation_times"]["rectangle_transformation"]
        )
        
        # Hold final state
        self.wait(config["animation_times"]["final_hold"])
        
        # === NEW SECTION: Add text, shift left, and create comparison ===
        
        # Step 5: Add text below the square
        text_line1 = Text(
            config["left_text"]["line1"],
            font_size=config["left_text"]["font_size"],
            color=config["left_text"]["color"],
            font=config["left_text"]["font"]
        )
        text_line2 = Text(
            config["left_text"]["line2"],
            font_size=config["left_text"]["font_size"],
            color=config["left_text"]["color"],
            font=config["left_text"]["font"]
        )
        
        # Position text below the current square
        text_group_center = config["rectangle"]["final_position"] + DOWN * (config["rectangle"]["final_width"]/2 + 0.8)
        text_line1.move_to(text_group_center + UP * config["left_text"]["spacing"]/2)
        text_line2.move_to(text_group_center + DOWN * config["left_text"]["spacing"]/2)
        
        # Animate text appearance
        self.play(
            Write(text_line1),
            Write(text_line2),
            run_time=config["animation_times"]["text_appear"]
        )
        
        # Step 6: Shift everything to the left
        # Note: initial_rectangle and initial_image have been transformed, so we work with them directly
        left_elements = Group(initial_rectangle, initial_image, text_line1, text_line2)
        
        self.play(
            left_elements.animate.shift(config["left_position"]),
            run_time=config["animation_times"]["shift_left"]
        )
        
        # Step 7: Create right side elements
        right_square = Rectangle(
            width=config["rectangle"]["final_width"],
            height=config["rectangle"]["final_width"],
            color=config["right_square"]["color"],
            stroke_width=config["right_square"]["stroke_width"],
            fill_opacity=config["right_square"]["fill_opacity"]
        )
        right_square.move_to(config["right_position"])
        
        right_image = ImageMobject(config["right_image"]["path"])
        right_image.scale_to_fit_width(config["rectangle"]["final_width"] * config["right_image"]["scale"])
        right_image.scale_to_fit_height(config["rectangle"]["final_width"] * config["right_image"]["scale"])
        right_image.move_to(config["right_position"])
        
        # Right side text (positioned at same vertical level as left text)
        right_text = Text(
            config["right_text"]["content"],
            font_size=config["right_text"]["font_size"],
            color=config["right_text"]["color"],
            font=config["right_text"]["font"]
        )
        # Position at same vertical level as left text group (accounting for left shift)
        left_text_y = text_group_center[1]
        right_text.move_to([config["right_position"][0], left_text_y, 0])
        
        # Animate right side elements appearance
        self.play(
            FadeIn(right_square),
            FadeIn(right_image),
            Write(right_text),
            run_time=config["animation_times"]["right_elements_appear"]
        )
        
        # Step 8: Add arrow from left to right
        arrow = Arrow(
            start=config["arrow"]["start"],
            end=config["arrow"]["end"],
            color=config["arrow"]["color"],
            stroke_width=config["arrow"]["stroke_width"],
            buff=0.5  # Space between arrow and elements
        )
        
        self.play(
            GrowArrow(arrow),
            run_time=config["animation_times"]["arrow_appear"]
        )
        
        # Step 9: Add question mark above the arrow
        question_mark = Text(
            config["question_mark"]["content"],
            font_size=config["question_mark"]["font_size"],
            color=config["question_mark"]["color"],
            font=config["question_mark"]["font"]
        )
        question_mark.move_to(arrow.get_center() + UP * 0.8)
        
        self.play(
            Write(question_mark),
            run_time=config["animation_times"]["question_appear"]
        )
        
        # Hold final extended state
        self.wait(config["animation_times"]["final_extended_hold"])
        
        # Step 10: Fade out everything
        all_elements = Group(
            initial_rectangle,  # Left square (transformed)
            initial_image,      # Left image (transformed)
            text_line1,         # "Multimode fiber"
            text_line2,         # "speckle pattern"
            right_square,       # Right square
            right_image,        # Right image
            right_text,         # "Original distribution"
            arrow,              # Arrow
            question_mark       # Question mark
        )
        
        self.play(
            FadeOut(all_elements),
            run_time=config["animation_times"]["fade_out_all"]
        )