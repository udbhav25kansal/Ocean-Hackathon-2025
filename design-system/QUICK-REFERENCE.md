# Bikini Bottom Color System - Quick Reference Card

## 🎨 Core Brand Colors

```css
--color-primary-400: #F7E21B;     /* Sponge Yellow */
--color-secondary-400: #3DB5FF;   /* Seafoam Blue */
--color-tertiary-400: #FF6F6F;    /* Coral */
--color-jellyfish-400: #C58CFF;   /* Lavender */
--color-seaweed-400: #2BB673;     /* Green */
--color-primary-contrast: #0B1E3A; /* Deep Navy */
```

## 🌈 Most Used Tokens

### Backgrounds
```css
--bg-primary         /* Light: #F6FBFD | Dark: #0B1E3A */
--surface-primary    /* Light: #FFFFFF | Dark: #112A4B */
```

### Text
```css
--text-primary       /* Light: #1E2A36 | Dark: #EAF2F7 */
--text-secondary     /* Light: #516171 | Dark: #B8C7D5 */
--text-link          /* Light: #1888D1 | Dark: #70D1FF */
```

### Interactive
```css
--interactive-primary        /* Light: #F7E21B | Dark: #FFED8C */
--interactive-primary-hover  /* Light: #E1C317 | Dark: #FFF3B3 */
--interactive-secondary      /* Light: #3DB5FF | Dark: #3DB5FF */
```

### Borders
```css
--border-primary     /* Light: #D7E3EC | Dark: #3A4752 */
--border-focus       /* Light: #3DB5FF | Dark: #70D1FF */
```

## ✅ Safe Text Combinations

### Light Mode
| Text Color | Background | Ratio |
|------------|------------|-------|
| `#0B1E3A` | `#FFFFFF` | 15.5:1 AAA ✓ |
| `#0B1E3A` | `#F7E21B` | 8.2:1 AAA ✓ |
| `#1E2A36` | `#FFFFFF` | 13.8:1 AAA ✓ |
| `#FFFFFF` | `#1888D1` | 4.5:1 AA ✓ |
| `#FFFFFF` | `#E63946` | 5.1:1 AA ✓ |

### Dark Mode
| Text Color | Background | Ratio |
|------------|------------|-------|
| `#EAF2F7` | `#0B1E3A` | 13.1:1 AAA ✓ |
| `#0B1E3A` | `#FFED8C` | 10.1:1 AAA ✓ |
| `#B8C7D5` | `#0B1E3A` | 8.9:1 AAA ✓ |
| `#70D1FF` | `#0B1E3A` | 6.8:1 AAA ✓ |

## 🎯 Component Recipes

### Primary Button
```css
.btn-primary {
  background: var(--interactive-primary);
  color: var(--text-on-primary);
}
.btn-primary:hover {
  background: var(--interactive-primary-hover);
}
```

### Success Alert
```css
.alert-success {
  background: var(--color-success-bg);
  border: 1px solid var(--color-success-border);
  color: var(--color-success);
}
```

### Focus Ring
```css
.input:focus-visible {
  outline: none;
  box-shadow: 0 0 0 2px var(--focus-ring-color);
}
```

### Gradient Hero
```css
.hero {
  background: var(--gradient-sunlit-water);
  color: var(--text-on-primary);
}
```

## 🚫 Don't Do This

❌ Yellow text on white: `#F7E21B` on `#FFFFFF` (1.9:1 FAIL)
❌ Coral text on navy: `#FF6F6F` on `#0B1E3A` (2.8:1 FAIL)
❌ Lavender text on white: `#C58CFF` on `#FFFFFF` (2.6:1 FAIL)

## ✅ Do This Instead

✓ Navy text on yellow: `#0B1E3A` on `#F7E21B` (8.2:1 AAA)
✓ White text on error red: `#FFFFFF` on `#E63946` (5.1:1 AA)
✓ Dark purple text on white: `#451A73` on `#FFFFFF` (7.8:1 AAA)

## 🌗 Toggle Dark Mode

```html
<html data-theme="light">
  <!-- or -->
<html data-theme="dark">
```

```javascript
// Toggle function
function toggleTheme() {
  const html = document.documentElement;
  const current = html.getAttribute('data-theme');
  html.setAttribute('data-theme', current === 'dark' ? 'light' : 'dark');
}
```

## 📦 Import

```html
<link rel="stylesheet" href="./design-system/colors.css">
```

Or use tokens:
```javascript
import tokens from './design-system/color-tokens.json';
```
