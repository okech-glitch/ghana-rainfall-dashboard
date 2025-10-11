#!/bin/bash

echo "🌧️ Rainfall Prediction Dashboard Deployment Script"
echo "=================================================="

# Check if vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

echo "✅ Vercel CLI found"

# Check if logged in to Vercel
if ! vercel whoami &> /dev/null; then
    echo "🔐 Please log in to Vercel:"
    vercel login
fi

echo "🚀 Deploying to Vercel..."

# Deploy to production
vercel --prod

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Set up custom domain (optional): vercel domain add <your-domain>"
echo "2. Add environment variables if needed: vercel env add FLASK_ENV"
echo "3. Monitor deployments: vercel ls"
echo ""
echo "🌐 Your dashboard will be available at: https://your-project.vercel.app"
echo ""
echo "💡 For portfolio: Add this project to showcase full-stack ML deployment!"
