@echo off
IF "%~1"=="all" (
    call :clean_release
    call :clean_debug
) ELSE IF "%~1"=="release" (
    call :clean_release
) ELSE IF "%~1"=="debug" (
    call :clean_debug
) ELSE (
    echo.
    echo clean.bat usage:
    echo clean release - cleans the release version
    echo clean debug - cleans the debug version
    echo clean all - cleans both versions
    echo NOTE -- This does not cleans the cmake cache
)
goto :End

:clean_release
    echo Cleaning Release
    IF EXIST "./build/release" (
        rmdir "./build/release" /s /q
    )
goto :EOF

:clean_debug
    echo Cleaning Debug
    IF EXIST "./build/debug" (
        rmdir "./build/debug" /s /q
    )
goto :EOF

:End