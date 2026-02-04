import discord
from discord.ext import commands
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='$', intents=intents)

try:
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    print("✅ Modelo cargado correctamente.")
except Exception as e:
    print(f"❌ Error crítico al cargar el modelo: {e}")

def get_class(image_path):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    
    prediction = model.predict(data)
    index = np.argmax(prediction)
    return class_names[index][2:].strip(), prediction[0][index]

@bot.event
async def on_ready():
    print(f'🚀 Bot listo: {bot.user}')


@bot.command()
async def hello(ctx):
    await ctx.send(f'Helooooooooooooo manolo, its me; estoy disponible para ayudarte en lo que quieras :).')

@bot.command()
async def heh(ctx, count_heh = 5):
    await ctx.send("he" * count_heh)

@bot.command()
async def check(ctx):
    """Implementación de Inferencia con manejo de errores"""
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            if not attachment.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                await ctx.send("❌ Error: Por favor sube una imagen (JPG o PNG).")
                continue

            try:
                file_path = attachment.filename
                await attachment.save(file_path)
                await ctx.send("🔎 Procesando...")
                
                clase, score = get_class(file_path)
                
                if clase.lower() == "nada" or score < 0.5:
                    await ctx.send("🤔 No estoy totalmente seguro que es lo de la imagen.")
                else:
                    await ctx.send(f"✅ Tarea completada: Es un(a) **{clase}** ({score*100:.2f}%)")
            
            except Exception as e:
                await ctx.send(f"⚠️ Hubo un error procesando la imagen: {e}")
    else:
        await ctx.send("Sube una imagen junto con el comando $check")

bot.run("UR_TOKEN_HERE")

