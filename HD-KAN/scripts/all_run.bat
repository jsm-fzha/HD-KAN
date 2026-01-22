@echo off

call scripts/LongForecasting/HDKAN.bat
call scripts/ShortForecasting/HDKAN_s1.bat
call scripts/ShortForecasting/HDKAN_s2.bat

echo All bat files finished.
pause
