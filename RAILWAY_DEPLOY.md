# Railway Deployment Guide
# Railway is perfect for Flask apps with generous free tier

# 1. Sign up at https://railway.app (free)

# 2. Install Railway CLI (optional, can use web interface)
npm install -g @railway/cli

# 3. Login to Railway
railway login

# 4. Initialize project (from web dashboard)
# Go to https://railway.app/new and select "Deploy from GitHub"
# Or use CLI: railway init

# 5. Deploy (Railway will auto-detect Flask and set up properly)
railway up

# Alternative: Deploy from GitHub
# 1. Push code to GitHub
# 2. Connect GitHub repo to Railway
# 3. Auto-deploy on every push

# Railway will provide a URL like: https://your-app.railway.app
