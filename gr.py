import gradio as gr
from train_gr import main

# Function to calculate the aspect ratio
def calculate_aspect_ratio(width, height):
    return width / height

# Function to update image dimensions based on aspect ratio
def update_dimensions(value, aspect_ratio, dim):
    if dim == "width":
        new_height = int(value / aspect_ratio)
        return gr.update(value=value, visible=True), gr.update(value=new_height, visible=True)
    elif dim == "height":
        new_width = int(value * aspect_ratio)
        return gr.update(value=new_width, visible=True), gr.update(value=value, visible=True)

# Function to handle image size selection
def handle_image_size_selection(img_size, content_img):
    if img_size == "custom size":
        height, width, _ = content_img.shape
        aspect_ratio = calculate_aspect_ratio(width, height)
        return (gr.update(value=width, visible=True),
                gr.update(value=height, visible=True),
                gr.update(visible=True), 
                gr.update(visible=True), 
                aspect_ratio)
    else:
        return (gr.update(value=450, visible=False),
                gr.update(value=300, visible=False),
                gr.update(visible=False), 
                gr.update(visible=False), 
                None)

# Define the function to process images
def process_images(content_img, style_img, epochs, steps_per_epoch, learning_rate, content_loss_factor, style_loss_factor, img_size, img_width, img_height):
    print("Start processing")
    output_img = main(content_img, style_img, epochs, steps_per_epoch, learning_rate, content_loss_factor, style_loss_factor, img_size, img_width, img_height)
    return output_img

with gr.Blocks() as demo:
    aspect_ratio = gr.State(None)
    with gr.Row():
        with gr.Column(scale=1):
            content_img = gr.Image(type="numpy", label="Content Image")
            style_img = gr.Image(type="numpy", label="Style Image")
            process_button = gr.Button("Process")
            with gr.Accordion("Parameters", open=False):
                epochs = gr.Slider(minimum=1, maximum=100, step=1, label="Epochs", value=20)
                steps_per_epoch = gr.Slider(minimum=1, maximum=1000, step=1, label="Steps per Epoch", value=100)
                learning_rate = gr.Slider(minimum=0.0001, maximum=0.1, step=0.0001, label="Learning Rate", value=0.01)
                content_loss_factor = gr.Slider(minimum=0.1, maximum=10, step=0.1, label="Content Loss Factor", value=1.0)
                style_loss_factor = gr.Slider(minimum=0.1, maximum=1000, step=0.1, label="Style Loss Factor", value=100.0)
                img_size = gr.Dropdown(choices=["default size", "custom size"], label="Image Size", value="default size")
                img_width = gr.Number(label="Image Width", value=450, visible=False)
                img_height = gr.Number(label="Image Height", value=300, visible=False)
        with gr.Column(scale=1):
            output_img = gr.Image(label="Output Image")

    img_size.change(
        fn=handle_image_size_selection,
        inputs=[img_size, content_img],
        outputs=[img_width, img_height, img_width, img_height, aspect_ratio]
    )

    img_width.change(
        fn=lambda w, ar: update_dimensions(w, ar, "width"),
        inputs=[img_width, aspect_ratio],
        outputs=[img_width, img_height]
    )

    img_height.change(
        fn=lambda h, ar: update_dimensions(h, ar, "height"),
        inputs=[img_height, aspect_ratio],
        outputs=[img_width, img_height]
    )

    process_button.click(
        process_images,
        inputs=[content_img, style_img, epochs, steps_per_epoch, learning_rate, content_loss_factor, style_loss_factor, img_size, img_width, img_height],
        outputs=[output_img]
    )

# Launch the app
demo.launch()