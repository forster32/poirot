import os
from typing import Literal, Iterator

from openai import OpenAI
from loguru import logger
from pydantic import BaseModel

from core.entities import Message
from core.prompts import system_message_prompt
from utils.str_utils import truncate_text_based_on_stop_sequence

# Importing DEFAULT_GPT4_MODEL from the server configuration
from config.server import DEFAULT_GPT4_MODEL
# Defining a type alias for OpenAI models
OpenAIModel = Literal[
  "gpt-3.5-turbo",
  "gpt-3.5-turbo-1106",
  "gpt-3.5-turbo-16k",
  "gpt-3.5-turbo-16k-0613",
  "gpt-4-1106-preview",
  "gpt-4-0125-preview",
  "gpt-4-turbo-2024-04-09",
  "gpt-4-turbo",
  "gpt-4o"
]

ChatModel = OpenAIModel

model_to_max_tokens = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-1106": 16385,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-4-1106-preview": 128000,
    "gpt-4-0125-preview": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    "gpt-4o": 128000,
    "gpt-3.5-turbo-16k-0613": 16000,
}

default_temperature = 0.1

class MessageList(BaseModel):
    messages: list[Message] = [
        Message(
            role="system",
            content=system_message_prompt,
        )
    ]

    @property
    def messages_dicts(self):
        # Remove the key from the message object before sending to OpenAI
        cleaned_messages = [message.to_openai() for message in self.messages]
        return cleaned_messages

class ChatGPT(MessageList):
    prev_message_states: list[list[Message]] = []
    model: ChatModel = DEFAULT_GPT4_MODEL

    def chat_openai(
        self,
        content: str,
        assistant_message_content: str = "",
        message_key: str | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] = [],
        stream: bool = False,
        max_tokens: int = 4096,
    ) -> str | Iterator[str]:
        if content:
            self.messages.append(Message(role="user", content=content, key=message_key))
        if assistant_message_content:
            self.messages.append(Message(role="assistant", content=assistant_message_content))
        temperature = temperature or self.temperature or default_temperature

        content = ""
        e = None

        try:
            client = OpenAI()
            
            print("Starting OpenAI stream")
            response = client.chat.completions.create(
                model=self.model,
                messages=self.messages_dicts,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            streamed_text = ""
            text = ""
            for chunk in response:
                new_content = chunk.choices[0].delta.content
                text += new_content if new_content else ""
                if new_content:
                    print(new_content, end="", flush=True)
                    streamed_text += new_content
                    for stop_sequence in stop_sequences:
                        if stop_sequence in streamed_text:
                            return truncate_text_based_on_stop_sequence(streamed_text, stop_sequences)
            print() # clear the line
            content = truncate_text_based_on_stop_sequence(streamed_text, stop_sequences)
        except Exception as e_:
            logger.exception(e_)

        self.messages.append(
            Message(
                role="assistant",
                content=content,
                key=message_key,
            )
        )
        self.prev_message_states.append(self.messages)
        return self.messages[-1].content
