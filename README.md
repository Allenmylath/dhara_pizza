# 🍕 Dhara Pizza

Voice-powered pizza ordering server built on [Rustvani](https://crates.io/crates/rustvani).

WebSocket voice agent with a Dhara conversation flow: **greeting → menu → confirm → farewell**.

## Pipeline

```
WebSocket → RAVI → Sarvam STT → LLM Aggregator → OpenAI GPT-4o-mini → TTS Aggregator → Deepgram TTS → WebSocket
```

Dhara manages conversation state and tool dispatch (browse menu, add/remove items, confirm, place order). Orders are validated against and persisted to a Neon PostgreSQL database.

## Setup

```bash
cp .env.example .env
# Fill in DATABASE_URL, SARVAM_API_KEY, OPENAI_API_KEY, DEEPGRAM_API_KEY
```

### Local

```bash
cargo run --release
# Server starts on ws://0.0.0.0:10000/ws
```

### Fly.io

```bash
fly launch          # first time
fly secrets set DATABASE_URL="..." SARVAM_API_KEY="..." OPENAI_API_KEY="..." DEEPGRAM_API_KEY="..."
fly deploy
```

## Database

Expects these tables in Neon (or any PostgreSQL):

- `pizzas` — id, name, description, is_vegetarian, image_url, is_available
- `pizza_sizes` — pizza_id, size, price
- `toppings` — name, price_per_unit, is_available
- `orders` — id, delivery_address, status, payment_completed, total_price
- `order_items` — order_id, pizza_id, pizza_name, size, extra_toppings, line_price

## License

MIT
