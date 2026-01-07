# Pano360GS - Technical Artist Guide

## Visual Overview

Este documento explica cómo funciona Pano360GS desde la perspectiva de un Technical Artist, con énfasis en los aspectos visuales y prácticos.

---

## 🌐 ¿Qué es una Proyección Equirectangular?

Una imagen equirectangular "desenrolla" una esfera completa en un rectángulo 2D.

```
         ┌─────────────────────────────────────────────────────────────┐
         │                      IMAGEN EQUIRECTANGULAR                  │
         │                        (ratio 2:1)                           │
         ├─────────────────────────────────────────────────────────────┤
         │                                                             │
v=0      │ ════════════════ POLO NORTE (mirando arriba) ═══════════════│
         │                                                             │
         │    ←── IZQUIERDA          FRENTE          DERECHA ──→       │
v=H/2    │         (-X)               (+Z)             (+X)            │
         │                                                             │
         │                        ← ATRÁS →                            │
         │                           (-Z)                              │
         │                                                             │
v=H      │ ════════════════ POLO SUR (mirando abajo) ══════════════════│
         └─────────────────────────────────────────────────────────────┘
              u=0                    u=W/2                          u=W
```

### Ejemplos de Ratios de Imagen

| Resolución | Ratio | ¿Es equirectangular? |
|------------|-------|---------------------|
| 4096 × 2048 | 2:1 | ✅ Sí |
| 3840 × 1920 | 2:1 | ✅ Sí |
| 2048 × 1024 | 2:1 | ✅ Sí |
| 1920 × 1080 | 16:9 | ❌ No (foto normal) |
| 1024 × 1024 | 1:1 | ❌ No (cuadrada) |

---

## 🎯 El Truco: Proyección Esférica

### Foto Normal (Pinhole) vs 360° (Esférica)

```
     FOTO NORMAL (PINHOLE)                    360° (ESFÉRICA)
     
           Cámara                                 Cámara
              │                                      ●
              │                                   ╱╲│╱╲
              ▼                                  ╱  │  ╲
         ╱─────────╲                            ╱   │   ╲
        ╱           ╲                          ╱    │    ╲
       ╱   Escena    ╲                        ╱     │     ╲
      ╱               ╲                      ╱      │      ╲
     ╱                 ╲                    ╱       │       ╲
    ╱___________________╲                  ╱_________________╲
    
    Los rayos salen en                   Los rayos salen en 
    forma de PIRÁMIDE                    forma de ESFERA
    (campo de visión limitado)           (360° completos)
```

---

## 🔧 Ajustes Automáticos

### 1. Compensación de Latitud

Los píxeles cerca de los polos representan áreas más pequeñas. Sin compensación, los polos se verían "vacíos":

```
           ANTES (sin compensación)              DESPUÉS (con compensación)
           
               ○ ○ ○                                 ● ● ●
             ○ ○ ○ ○ ○                             ● ● ● ● ●
            ○ ○ ○ ○ ○ ○ (ecuador lleno)           ● ● ● ● ● ●
             ○ ○ ○ ○ ○                             ● ● ● ● ●
               ○ ○ ○                                 ● ● ●
               
           Menos splats                          Splats más grandes
           en los polos                          compensan la densidad
```

### 2. Escala por Profundidad

Objetos lejanos necesitan splats más grandes para no dejar huecos:

```
    Cercano (depth=1m)          Lejano (depth=10m)
    
         ●                           ●●●
        ● ●                        ●●●●●
         ●                         ●●●●●
                                    ●●●
    
    Splats pequeños              Splats grandes
    (detalle fino)               (mantiene cobertura)
```

### 3. Rotación Tangente

Cada splat se rota para ser tangente a la esfera:

```
           Vista lateral del splat
           
    En imagen plana:              En esfera:
    
         ▓▓▓▓▓                         ╱▓▓▓▓╲
         ▓▓▓▓▓                        ╱▓▓▓▓▓▓╲
         ▓▓▓▓▓                       ●──▓▓▓▓──●
    (orientación fija)                ╲▓▓▓▓▓▓╱
                                       ╲▓▓▓▓╱
                                   (tangente a esfera)
```

---

## 🎨 Niveles de Calidad

Visualización del impacto de cada nivel:

```
    LOW (~500K splats)          MEDIUM (~2M splats)
    ┌─────────────────┐         ┌─────────────────┐
    │ ● ● ● ● ● ● ● ●│         │●●●●●●●●●●●●●●●●│
    │ ● ● ● ● ● ● ● ●│         │●●●●●●●●●●●●●●●●│
    │ ● ● ● ● ● ● ● ●│         │●●●●●●●●●●●●●●●●│
    └─────────────────┘         └─────────────────┘
    Rápido, preview             Balance calidad/rendimiento
    
    HIGH (~5M splats)           ULTRA (~8.4M splats)
    ┌─────────────────┐         ┌─────────────────┐
    │█████████████████│         │█████████████████│
    │█████████████████│         │█████████████████│
    │█████████████████│         │█████████████████│
    └─────────────────┘         └─────────────────┘
    Alta calidad                Máxima calidad
                                (GPU potente requerida)
```

### Recomendaciones por Plataforma

| Plataforma | Calidad Recomendada | Notas |
|------------|---------------------|-------|
| Web (móvil) | LOW | Ancho de banda limitado |
| Web (desktop) | MEDIUM | Buen balance |
| Quest 2/3 | MEDIUM-HIGH | Depende de complejidad |
| PC VR | HIGH-ULTRA | Hardware potente |
| Pre-renderizado | ULTRA | Sin límite de tiempo real |

---

## 📐 Sistema de Coordenadas

Pano360GS usa convención **Y-up** (compatible con Unity, Unreal, OpenCV):

```
                    +Y (arriba)
                     │
                     │
                     │
                     │
    ─────────────────┼─────────────────→ +X (derecha)
                    ╱│
                   ╱ │
                  ╱  │
                 ╱   │
                ↙    │
              +Z (adelante - hacia donde mira el centro de la imagen)
```

### Mapeo de Direcciones en la Imagen

| Posición en imagen | Ángulos | Dirección 3D |
|-------------------|---------|--------------|
| Centro (W/2, H/2) | φ=0, θ=0 | +Z (adelante) |
| Borde izquierdo | φ=-π, θ=0 | -Z (atrás) |
| Borde derecho | φ=+π, θ=0 | -Z (atrás) |
| Arriba (cualquier x) | θ=+π/2 | +Y (arriba) |
| Abajo (cualquier x) | θ=-π/2 | -Y (abajo) |
| 1/4 desde izquierda | φ=-π/2, θ=0 | -X (izquierda) |
| 3/4 desde izquierda | φ=+π/2, θ=0 | +X (derecha) |

---

## 🔍 Verificación Visual

### Checklist de Calidad

- [ ] ¿El visor muestra contenido en las 6 direcciones?
      (arriba, abajo, adelante, atrás, izquierda, derecha)
- [ ] ¿Los polos tienen splats visibles (no vacíos)?
- [ ] ¿Las transiciones entre áreas son suaves?
- [ ] ¿Los objetos lejanos mantienen cobertura?
- [ ] ¿No hay artefactos de "costura" en φ=±π (donde se unen los bordes)?

### Visores Recomendados para Testing

1. **Web**: [antimatter15/splat](https://antimatter15.com/splat/)
2. **Unity**: [com.witalosk.unity-gaussian-splatting](https://github.com/witalosk/UnityGaussianSplatting)
3. **Unreal**: Native 3DGS support (UE 5.4+)

---

## 💡 Tips para Technical Artists

### Preparación de Imágenes

1. **Resolución recomendada**: 4096×2048 o superior
2. **Formato**: JPG o PNG (sin compresión extrema)
3. **Verificar ratio**: Siempre 2:1

### Optimización del Output

1. **Filtrar splats de baja opacidad** después de generar:
   ```python
   mask = gaussians.opacities > 0.01
   # Aplicar máscara para reducir conteo de splats
   ```

2. **Ajustar escala global** si los splats parecen muy grandes/pequeños:
   ```python
   gaussians.singular_values *= 0.8  # Reducir 20%
   ```

### Debug de Problemas Comunes

| Síntoma | Causa Probable | Solución |
|---------|----------------|----------|
| Escena invertida | Y-axis flip | Negar coordenada Y |
| Huecos en los polos | Falta compensación | Verificar `latitude_compensation=True` |
| Splats gigantes en polos | Compensación excesiva | Reducir clamp máximo |
| Escena muy pequeña/grande | Escala de profundidad | Ajustar `depth_scale_factor` |

---

## 🎬 Workflow Típico

```
┌──────────────────┐
│ 1. Captura 360°  │
│    o genera con  │
│    FLUX/Midjourney│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 2. Verificar     │
│    ratio 2:1     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 3. Procesar con  │
│    Pano360GS     │
│    (elegir       │
│     calidad)     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 4. Exportar .PLY │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 5. Importar en   │
│    Unity/Unreal  │
│    o visor web   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 6. Ajustar       │
│    cámara al     │
│    origen (0,0,0)│
└──────────────────┘
```

---

*Documento preparado para Technical Artists trabajando con Pano360GS*
*Última actualización: Enero 2025*
