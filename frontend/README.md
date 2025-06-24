## 🖼️ Frontend – Next.js UI

This folder contains the **Next.js** application that acts as a user-friendly interface for the FastAPI backend found in `/api`.

### 🚀 Getting Started Locally

1. **Install dependencies** (make sure you are in the `frontend` directory):

   ```bash
   cd frontend
   npm install
   ```

2. **Run the development server**:

   ```bash
   npm run dev
   ```

   The app will be available at **http://localhost:3000**.

3. **Use the UI**:

   * Enter a *Developer* message (system instructions)
   * Enter a *User* message
   * Paste your **OpenAI API key** (the input is a password field and will be hidden)
   * Optionally override the model name (defaults to `gpt-4.1-mini`)
   * Click **Send** and watch the streamed response appear in real-time 🎉

### 🏗️ Building for Production

```bash
npm run build        # generates the production build
npm start            # starts the Next.js server
```

### ☁️ Deploying with Vercel

The project root already contains a `vercel.json` that configures Vercel to build the frontend and route API requests to the FastAPI backend.

1. Install the Vercel CLI if you haven't already:

   ```bash
   npm install -g vercel
   ```

2. From the repository root, run:

   ```bash
   vercel
   ```

   Follow the prompts and your full-stack app (frontend **and** backend) will be live in minutes 🌐🚀.

---

Made with ❤️ for the AI Engineer Challenge.