from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    filters,
    CommandHandler,
    ContextTypes,
)
import cv2
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import logging
import os

""" # Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
) """

# Replace with your bot token
TOKEN = os.getenv("TOKEN")
FONT_PATH = "Trap-ExtraBold.otf"
LETTER_SPACING = -3.5
TEXT = "waa"
TEXT_COVERAGE = 0.4


async def add_text(image, text=TEXT):
    # Create a transparent overlay for the text
    text_overlay = Image.new("RGBA", (image.shape[1], image.shape[0]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_overlay)

    # Find appropriate font size
    font_size = 1
    font = ImageFont.truetype(FONT_PATH, font_size)
    text_width = sum(draw.textlength(char, font=font) for char in text) + (
        len(text) - 1
    ) * (draw.textlength("a", font=font) * LETTER_SPACING / 100)

    target_width = image.shape[1] * TEXT_COVERAGE

    while text_width < target_width:
        font_size += 1
        font = ImageFont.truetype(FONT_PATH, font_size)
        text_width = sum(draw.textlength(char, font=font) for char in text) + (
            len(text) - 1
        ) * (draw.textlength("a", font=font) * LETTER_SPACING / 100)

    # Calculate text position
    width, height = text_overlay.size
    total_width = text_width
    current_x = (width - total_width) // 2
    y = height // 2

    # Draw semi-transparent text
    for i, char in enumerate(text):
        bbox = draw.textbbox((0, 0), char, font=font)
        char_width = bbox[2] - bbox[0]
        draw.text(
            (current_x, y),
            char,
            font=font,
            # Set opacity amount here NIGGA
            fill=(0, 0, 0, round(256 * 0.85)),
            anchor="lm",
        )

        if i < len(text) - 1:
            spacing = char_width * (LETTER_SPACING / 100)
            current_x += char_width + spacing

    # Convert background image to PIL
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Composite the images
    result = Image.alpha_composite(image_pil.convert("RGBA"), text_overlay)
    result = result.convert("RGB")

    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


async def apply_filter(photo, context):
    file = await context.bot.get_file(photo.file_id)
    image_bytes = await file.download_as_bytearray()

    image_stream = BytesIO(image_bytes)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    image_float = image.astype(float)
    image_float[:, :, 0] *= 0.75
    image_float[:, :, 2] *= 0.75
    image_float[:, :, 1] = np.minimum(image_float[:, :, 1] * 1.6 + 50, 255)

    noise = np.random.normal(0, 5, image.shape)
    image_float += noise

    filtered_image = np.clip(image_float, 0, 255).astype(np.uint8)
    final_image = await add_text(filtered_image)

    success, buffer = cv2.imencode(".jpg", final_image)
    bio = BytesIO(buffer)
    bio.seek(0)
    return bio


async def handle_waaify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    logging.debug(f"Full message object: {message}")

    if not message.reply_to_message:
        logging.debug("No reply_to_message")
        return

    reply = message.reply_to_message
    logging.debug(f"Reply message type: {type(reply)}")
    logging.debug(f"Reply message content: {reply}")

    if hasattr(reply, "photo"):
        photos = reply.photo
        logging.debug(f"Photos in reply: {photos}")
        if photos:
            try:
                photo = photos[-1]
                filtered_photo = await apply_filter(photo, context)
                await update.message.reply_photo(
                    photo=filtered_photo, reply_to_message_id=reply.message_id
                )
            except Exception as e:
                logging.error(f"Processing error: {str(e)}", exc_info=True)
                await message.reply_text(f"Error: {str(e)}")
    else:
        logging.debug("No photo attribute in reply")


def main():
    application = Application.builder().token(TOKEN).build()

    # Register command handler
    application.add_handler(CommandHandler("waaify", handle_waaify))

    # Log startup
    print("Bot started! Press Ctrl+C to stop")
    print("Registered command: /waaify")

    application.run_polling()


if __name__ == "__main__":
    main()
