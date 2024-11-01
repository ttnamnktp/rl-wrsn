

# Thêm tất cả các thay đổi vào staging
git add .

# Commit với message bao gồm ngày và giờ hiện tại
$commitMessage = "Auto commit on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
git commit -m "$commitMessage"

# Push thay đổi lên nhánh 'lam'
git push origin lam