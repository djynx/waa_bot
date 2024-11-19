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
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    filename="telegram_bot.log",  # Add file logging
)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TOKEN")
FONT_PATH = "Trap-ExtraBold.otf"
LETTER_SPACING = -3.5
TEXT = "waa"
TEXT_COVERAGE = 0.4

executor = ThreadPoolExecutor(max_workers=5)


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
    """Handle the /waaify command with proper error handling and user feedback."""
    try:
        message = update.message
        if not message.reply_to_message:
            await message.reply_text("Please reply to an image with /waaify")
            return

        reply = message.reply_to_message
        if not hasattr(reply, "photo") or not reply.photo:
            await message.reply_text("The replied message doesn't contain an image")
            return

        # Send a processing message
        processing_msg = await message.reply_text("Processing image... 🔄")

        try:
            photo = reply.photo[-1]
            filtered_photo = await apply_filter(photo, context)
            await update.message.reply_photo(
                photo=filtered_photo, reply_to_message_id=reply.message_id
            )
        finally:
            # Clean up processing message
            await processing_msg.delete()

    except Exception as e:
        error_message = f"Error processing image: {str(e)}"
        logger.error(error_message, exc_info=True)
        await message.reply_text(
            "Sorry, something went wrong while processing the image. Please try again later."
        )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors in the telegram-python-bot library."""
    logger.error(
        f"Update {update} caused error {context.error}", exc_info=context.error
    )


def main():
    try:
        # Build application with proper error handling
        application = (
            Application.builder()
            .token(TOKEN)
            .concurrent_updates(True)  # Enable concurrent updates
            .build()
        )

        # Register handlers
        application.add_handler(CommandHandler("waaify", handle_waaify))
        application.add_error_handler(error_handler)

        # Log startup
        logger.info("Bot started! Press Ctrl+C to stop")
        logger.info("Registered command: /waaify")

        # Start the bot with proper shutdown handling
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,  # Ignore any pending updates
        )

    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
