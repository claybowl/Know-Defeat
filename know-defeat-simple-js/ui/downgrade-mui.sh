#!/bin/bash
# This script downgrades MUI packages to versions that are compatible with our current Grid implementation

# Uninstall current versions
npm uninstall @mui/material @mui/icons-material @emotion/react @emotion/styled

# Install specific versions with legacy-peer-deps to bypass the React version check
npm install @mui/material@5.15.10 @mui/icons-material@5.15.10 @emotion/react@11.11.3 @emotion/styled@11.11.0 --legacy-peer-deps