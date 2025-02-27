import google.generativeai as genai
import redis
from PIL import Image

import io
import os
import requests
from io import BytesIO

from Lib import BotController, QQRichText
from google import genai
from google.genai import types
r = redis.Redis(host='localhost', port=6379, db=0)
api_key = r.get('llm_api_key')
client = genai.Client(api_key=api_key)
model_name = "gemini-2.0-flash-exp"
bounding_box_system_instructions = """
    Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
    If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
      """
safety_settings = [
  types.SafetySetting(
    category="HARM_CATEGORY_DANGEROUS_CONTENT",
    threshold="BLOCK_NONE",
  ),
]


# @title Plotting Util

# Get Noto JP font to display janapese characters
# For Noto Sans CJK JP
# @title Parsing JSON output
def parse_json(json_output):
  # Parsing out the markdown fencing
  lines = json_output.splitlines()
  for i, line in enumerate(lines):
    if line == "```json":
      json_output = "\n".join(lines[i + 1:])  # Remove everything before "```json"
      json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
      break  # Exit the loop once "```json" is found
  return json_output


# !apt-get install fonts-source-han-sans-jp # For Source Han Sans (Japanese)

import json
import random
import io
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]


def plot_bounding_boxes(im, bounding_boxes):
  """
  Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

  Args:
      img_path: The path to the image file.
      bounding_boxes: A list of bounding boxes containing the name of the object
       and their positions in normalized [y1 x1 y2 x2] format.
  Returns:
      str: The path to the saved image with bounding boxes
  """

  # Load the image
  img = im
  width, height = img.size
  print(img.size)
  # Create a drawing object
  draw = ImageDraw.Draw(img)

  # Define a list of colors
  colors = [
             'red',
             'green',
             'blue',
             'yellow',
             'orange',
             'pink',
             'purple',
             'brown',
             'gray',
             'beige',
             'turquoise',
             'cyan',
             'magenta',
             'lime',
             'navy',
             'maroon',
             'teal',
             'olive',
             'coral',
             'lavender',
             'violet',
             'gold',
             'silver',
           ] + additional_colors

  # Parsing out the markdown fencing
  bounding_boxes = parse_json(bounding_boxes)

  font = ImageFont.load_default()

  # Iterate over the bounding boxes
  for i, bounding_box in enumerate(json.loads(bounding_boxes)):
    # Select a color from the list
    color = colors[i % len(colors)]

    # Convert normalized coordinates to absolute coordinates
    abs_y1 = int(bounding_box["box_2d"][0] / 1000 * height)
    abs_x1 = int(bounding_box["box_2d"][1] / 1000 * width)
    abs_y2 = int(bounding_box["box_2d"][2] / 1000 * height)
    abs_x2 = int(bounding_box["box_2d"][3] / 1000 * width)

    if abs_x1 > abs_x2:
      abs_x1, abs_x2 = abs_x2, abs_x1

    if abs_y1 > abs_y2:
      abs_y1, abs_y2 = abs_y2, abs_y1

    # Draw the bounding box
    draw.rectangle(
      ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
    )

    # Draw the text
    if "label" in bounding_box:
      draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

  # Save the image
  img.show()
  output_path = "output_bbox.png"  # 你可以修改保存路径和文件名
  img.save(output_path)
  return output_path
import io
import re
from io import BytesIO
from typing import Tuple, List, Optional
import os

import requests
from PIL import Image


import re

import re


def extract_image_url(cq_code: str) -> str:
  """
  Extracts the URL from a CQ code containing an image.

  :param cq_code: The CQ code string that contains the image URL.
  :return: The extracted URL or None if no URL is found.
  """
  # Regular expression to match the CQ:image tag and capture the URL attribute
  image_url_pattern = re.compile(r'\[CQ:image.*?url=([^,\]]+)')

  # Search for the URL within the CQ code
  match = image_url_pattern.search(cq_code)

  if match:
    # The URL is captured as the first group in the regex match
    url = match.group(1)

    # Clean up ampersand encoding (&amp; -> &)
    clean_url = url.replace("&amp;", "&")

    return clean_url
  else:
    # No URL found in the CQ code
    return None


def imageLLM(path, TargetPrompt,group_id):
  image = path
  prompt = f"Detect the 2d bounding boxes of the {TargetPrompt} in English.(with “label” as topping description”)"  # @param {type:"string"}

  # Load and resize image
  img = Image.open(BytesIO(open(image, "rb").read()))
  im = Image.open(image).resize((1024, int(1024 * img.size[1] / img.size[0])), Image.Resampling.LANCZOS)

  # Run model to find bounding boxes
  response = client.models.generate_content(
    model=model_name,
    contents=[prompt, im],
    config=types.GenerateContentConfig(
      system_instruction=bounding_box_system_instructions,
      temperature=0.5,
      safety_settings=safety_settings,
    )
  )

  # Check output
  print(response.text)
  # 在获取response后添加:
  BotController.send_message("test", group_id=group_id)
  formatted_path = os.path.abspath(plot_bounding_boxes(im, response.text)).replace('\\', '\\\\')
  BotController.send_message(QQRichText.Image(formatted_path), group_id=group_id)




def transLLM(path,TargetPrompt,group_id):
  image = path
  # Load and resize image
  img = Image.open(BytesIO(open(image, "rb").read()))
  im = Image.open(image).resize((1024, int(1024 * img.size[1] / img.size[0])), Image.Resampling.LANCZOS)
  SYSPROMPOT = "你需要将图片内的的对话翻译为中文，尤其对话框内容，原始文字可能是中文或者英文的，你需要注意英语，日语和需要翻译为的中文在习惯用词，俚语，主谓宾形式上的差距，将翻译内容的格式和风格更贴近中文语境，并且你需要通过翻译获得的文本有效的按照逻辑组织每句话之间的顺序，最后输出逐行按顺序翻译为中文的结果"

  # Run model to find bounding boxes
  response = client.models.generate_content(
    model=model_name,
    contents=[TargetPrompt, im],
    config=types.GenerateContentConfig(
      system_instruction=SYSPROMPOT,
      temperature=0.5,
      safety_settings=safety_settings,
    )
  )
  # Check output
  print(response.text)
  # 在获取response后添加:
  BotController.send_message(response.text, group_id=group_id)
