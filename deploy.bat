@echo off
echo 🌧️ Rainfall Prediction Dashboard Deployment Script
echo ==================================================
echo.

REM Check if vercel CLI is installed
vercel --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Vercel CLI not found. Installing...
    npm install -g vercel
) else (
    echo ✅ Vercel CLI found
)

echo.
echo 🔐 Checking Vercel login status...
vercel whoami >nul 2>&1
if errorlevel 1 (
    echo Please log in to Vercel:
    vercel login
)

echo.
echo 🚀 Deploying to Vercel...
vercel --prod

echo.
echo 🎉 Deployment complete!
echo.
echo 📋 Next steps:
echo 1. Set up custom domain ^(optional^): vercel domain add ^<your-domain^>
echo 2. Add environment variables if needed: vercel env add FLASK_ENV
echo 3. Monitor deployments: vercel ls
echo.
echo 🌐 Your dashboard will be available at: https://your-project.vercel.app
echo.
echo 💡 For portfolio: Add this project to showcase full-stack ML deployment!
pause
