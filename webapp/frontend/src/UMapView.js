import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as d3 from 'd3';
import { useTheme } from './ThemeProvider'; // Import the theme hook

// Color Legend Component
function ColorLegend({ continentColors, className = '' }) {
  return (
    <div className={`color-legend ${className}`}>
      <div className="legend-title">Continents</div>
      <div className="legend-items">
        {Object.entries(continentColors).map(([continent, color]) => (
          <div key={continent} className="legend-item">
            <div 
              className="legend-color" 
              style={{ backgroundColor: color }}
            ></div>
            <span className="legend-label">{continent}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Updated UMAP Component with proper theme integration
const UMapView = ({ locations, selectedLocations, cityFilteredLocations, onLocationSelect }) => {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [umapData, setUmapData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  
  // Use the proper theme hook instead of manual detection
  const { isDark, theme } = useTheme();
  
  // Performance optimization refs
  const animationFrameRef = useRef(null);
  const quadtreeRef = useRef(null);
  const scalesRef = useRef({ xScale: null, yScale: null });
  const hoveredPointRef = useRef(null);
  const lastMousePosRef = useRef({ x: 0, y: 0 });
  const isDraggingRef = useRef(false);
  const dragStartTimeRef = useRef(null);
  const dragDistanceRef = useRef(0);
  const transformRef = useRef({ k: 1, x: 0, y: 0 });
  const mouseDownPosRef = useRef({ x: 0, y: 0 });
  const resizeObserverRef = useRef(null);

  const API_BASE_URL = process.env.NODE_ENV === 'production' 
    ? process.env.REACT_APP_API_URL || 'http://localhost:8000'
    : 'http://localhost:8000';

  // Continent color mapping
  const continentColors = {
    'Africa': '#e74c3c',
    'Asia': '#3498db', 
    'Europe': '#2ecc71',
    'North America': '#f39c12',
    'South America': '#9b59b6',
    'Oceania': '#95a5a6',
    'Australia': '#95a5a6'
  };

  // Plot margins to ensure axes don't overlap with points
  const PLOT_MARGIN = { top: 30, right: 30, bottom: 60, left: 60 };

  // Enhanced thresholds for distinguishing click from drag
  const DRAG_THRESHOLD_PX = 5;
  const DRAG_THRESHOLD_TIME = 150;

  // Get theme-appropriate outline color - directly from DOM to avoid stale closures
  const getOutlineColor = useCallback(() => {
    // Always get fresh theme state directly from DOM classes to avoid stale closures during animations
    const isDarkMode = document.documentElement.classList.contains('dark-theme');
    return isDarkMode ? '#5a5866' : '#fff';
  }, []);

  // Force re-render when theme changes - this is the key fix
  useEffect(() => {
    if (umapData && canvasRef.current && scalesRef.current.xScale) {
      // Force immediate re-render with current theme
      const ctx = canvasRef.current.getContext('2d');
      renderPoints(ctx, umapData.umap_points, transformRef.current);
    }
  }, [isDark]); // React to theme changes immediately

  // Enhanced resize handler with ResizeObserver for better performance
  const handleResize = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    
    animationFrameRef.current = requestAnimationFrame(() => {
      const container = containerRef.current;
      if (container) {
        const rect = container.getBoundingClientRect();
        const newDimensions = {
          width: Math.max(400, rect.width),
          height: Math.max(300, rect.height)
        };
        
        // Only update if dimensions actually changed
        if (newDimensions.width !== dimensions.width || newDimensions.height !== dimensions.height) {
          setDimensions(newDimensions);
        }
      }
    });
  }, [dimensions.width, dimensions.height]);

  // Set up ResizeObserver for better resize detection
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Initial size calculation
    handleResize();

    // Set up ResizeObserver for automatic canvas resizing
    if (window.ResizeObserver) {
      resizeObserverRef.current = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const { width, height } = entry.contentRect;
          const newDimensions = {
            width: Math.max(400, width),
            height: Math.max(300, height)
          };
          
          if (newDimensions.width !== dimensions.width || newDimensions.height !== dimensions.height) {
            setDimensions(newDimensions);
          }
        }
      });
      
      resizeObserverRef.current.observe(container);
    } else {
      // Fallback to window resize event
      window.addEventListener('resize', handleResize);
    }

    return () => {
      if (resizeObserverRef.current) {
        resizeObserverRef.current.disconnect();
      } else {
        window.removeEventListener('resize', handleResize);
      }
      
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [handleResize]);

  // Fetch UMAP data when locations are available
  useEffect(() => {
    if (locations.length > 0 && !umapData) {
      fetchUmapData();
    }
  }, [locations]);

  // Listen for highlight events and zoom events
  useEffect(() => {
    const handleHighlightPoint = (event) => {
      const { locationId } = event.detail;
      highlightPoint(locationId);
    };

    const handleZoomToPoint = (event) => {
      const { locationId } = event.detail;
      zoomToPoint(locationId);
    };

    const handleCenterOnCityTiles = (event) => {
      const { locationIds } = event.detail;
      centerOnCityTiles(locationIds);
    };

    window.addEventListener('highlightUmapPoint', handleHighlightPoint);
    window.addEventListener('zoomToUmapPoint', handleZoomToPoint);
    window.addEventListener('centerOnCityTiles', handleCenterOnCityTiles);
    
    return () => {
      window.removeEventListener('highlightUmapPoint', handleHighlightPoint);
      window.removeEventListener('zoomToUmapPoint', handleZoomToPoint);
      window.removeEventListener('centerOnCityTiles', handleCenterOnCityTiles);
    };
  }, [umapData]);

  const fetchUmapData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/umap`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch UMAP data');
      }
      const data = await response.json();
      setUmapData(data);
    } catch (err) {
      setError(`Failed to load UMAP visualization: ${err.message}`);
      
      // Fallback: compute simple projection
      await computeSimpleProjection();
    } finally {
      setLoading(false);
    }
  };

  const computeSimpleProjection = async () => {
    try {
      const data = locations.map((location, index) => {
        const geoProjX = (location.longitude + 180) / 360;
        const geoProjY = (location.latitude + 90) / 180;
        
        const noise = () => (Math.random() - 0.5) * 0.3;
        
        return {
          location_id: location.id,
          x: geoProjX + noise(),
          y: geoProjY + noise(),
          city: location.city,
          country: location.country,
          continent: location.continent,
          longitude: location.longitude,
          latitude: location.latitude,
          date: location.date
        };
      });

      setUmapData({ 
        umap_points: data,
        total_points: data.length
      });
    } catch (err) {
      setError('Failed to compute UMAP visualization');
    }
  };

  // Build spatial index for fast hover detection
  const buildQuadtree = useCallback((data) => {
    if (!data || !scalesRef.current.xScale) return null;
    
    const { xScale, yScale } = scalesRef.current;
    
    return d3.quadtree()
      .x(d => {
        const baseX = xScale(d.x);
        return baseX * transformRef.current.k + transformRef.current.x;
      })
      .y(d => {
        const baseY = yScale(d.y);
        return baseY * transformRef.current.k + transformRef.current.y;
      })
      .addAll(data);
  }, []);

  // Function to rebuild quadtree when transform changes
  const rebuildQuadtree = useCallback(() => {
    if (umapData?.umap_points) {
      quadtreeRef.current = buildQuadtree(umapData.umap_points);
    }
  }, [umapData, buildQuadtree]);

  // Draw persistent axes that don't transform
  const drawAxes = useCallback((ctx) => {
    if (!scalesRef.current.xScale) return;
    
    const { xScale, yScale } = scalesRef.current;
    
    ctx.save();
    
    // Theme-aware axis colors - get fresh theme state directly from DOM to avoid stale closures
    const isDarkMode = document.documentElement.classList.contains('dark-theme');
    const axisColor = isDarkMode ? '#5a5866' : '#dee2e6';
    const textColor = isDarkMode ? '#a09db0' : '#666';
    
    ctx.strokeStyle = axisColor;
    ctx.fillStyle = textColor;
    ctx.lineWidth = 1;
    ctx.font = '11px sans-serif';
    
    const plotWidth = dimensions.width - PLOT_MARGIN.left - PLOT_MARGIN.right;
    const plotHeight = dimensions.height - PLOT_MARGIN.top - PLOT_MARGIN.bottom;
    
    // Draw axis lines
    ctx.beginPath();
    // X-axis (bottom)
    ctx.moveTo(PLOT_MARGIN.left, PLOT_MARGIN.top + plotHeight);
    ctx.lineTo(PLOT_MARGIN.left + plotWidth, PLOT_MARGIN.top + plotHeight);
    // Y-axis (left)
    ctx.moveTo(PLOT_MARGIN.left, PLOT_MARGIN.top);
    ctx.lineTo(PLOT_MARGIN.left, PLOT_MARGIN.top + plotHeight);
    ctx.stroke();
    
    // Draw tick marks and labels
    const xTicks = xScale.ticks(5);
    const yTicks = yScale.ticks(5);
    
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    xTicks.forEach(tick => {
      const x = xScale(tick);
      if (x >= PLOT_MARGIN.left && x <= PLOT_MARGIN.left + plotWidth) {
        // Tick mark
        ctx.beginPath();
        ctx.moveTo(x, PLOT_MARGIN.top + plotHeight);
        ctx.lineTo(x, PLOT_MARGIN.top + plotHeight + 5);
        ctx.stroke();
        // Label
        ctx.fillText(tick.toFixed(1), x, PLOT_MARGIN.top + plotHeight + 8);
      }
    });
    
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    yTicks.forEach(tick => {
      const y = yScale(tick);
      if (y >= PLOT_MARGIN.top && y <= PLOT_MARGIN.top + plotHeight) {
        // Tick mark
        ctx.beginPath();
        ctx.moveTo(PLOT_MARGIN.left - 5, y);
        ctx.lineTo(PLOT_MARGIN.left, y);
        ctx.stroke();
        // Label
        ctx.fillText(tick.toFixed(1), PLOT_MARGIN.left - 8, y);
      }
    });
    
    // Draw axis labels
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('UMAP Dimension 1', PLOT_MARGIN.left + plotWidth / 2, dimensions.height - 15);
    
    ctx.save();
    ctx.translate(15, PLOT_MARGIN.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('UMAP Dimension 2', 0, 0);
    ctx.restore();
    
    ctx.restore();
  }, [dimensions]);

  // Enhanced point rendering function with proper theme integration
  const renderPoints = useCallback((ctx, data, transform = { k: 1, x: 0, y: 0 }) => {
    if (!data || !scalesRef.current.xScale) return;
    
    const { xScale, yScale } = scalesRef.current;
    const { k, x: tx, y: ty } = transform;
    
    // Clear canvas
    ctx.clearRect(0, 0, dimensions.width, dimensions.height);
    
    // Draw axes first (they don't transform)
    drawAxes(ctx);
    
    // Render points with layering - city filtered -> selected -> hovered
    ctx.save();
    
    const plotWidth = dimensions.width - PLOT_MARGIN.left - PLOT_MARGIN.right;
    const plotHeight = dimensions.height - PLOT_MARGIN.top - PLOT_MARGIN.bottom;
    
    // Clip to plot area
    ctx.beginPath();
    ctx.rect(PLOT_MARGIN.left, PLOT_MARGIN.top, plotWidth, plotHeight);
    ctx.clip();
    
    // Separate points into layers
    const regularPoints = [];
    const cityFilteredPoints = [];
    const selectedPoints = [];
    const hoveredPoint = [];
    
    data.forEach(point => {
      const baseX = xScale(point.x);
      const baseY = yScale(point.y);
      
      // Apply transform manually for each point
      const x = baseX * k + tx;
      const y = baseY * k + ty;
      
      // Only process visible points
      if (x < PLOT_MARGIN.left - 20 || x > dimensions.width - PLOT_MARGIN.right + 20 || 
          y < PLOT_MARGIN.top - 20 || y > dimensions.height - PLOT_MARGIN.bottom + 20) {
        return;
      }
      
      const pointData = { ...point, screenX: x, screenY: y };
      
      if (hoveredPointRef.current === point.location_id) {
        hoveredPoint.push(pointData);
      } else if (selectedLocations.has(point.location_id)) {
        selectedPoints.push(pointData);
      } else if (cityFilteredLocations.has(point.location_id)) {
        cityFilteredPoints.push(pointData);
      } else {
        regularPoints.push(pointData);
      }
    });
    
    // Get CURRENT theme outline color directly - use fresh theme state each render
    const currentOutlineColor = getOutlineColor();
    
    // Render in layers: regular -> city filtered -> selected -> hovered
    const renderLayer = (points, isSelected = false, isHovered = false, isCityFiltered = false) => {
      points.forEach(point => {
        // REDUCED BASE SIZES - Made all points smaller
        const baseRadius = isSelected ? 4 : (isCityFiltered ? 3.5 : 2.5);
        const radius = isHovered ? baseRadius + 1.5 : baseRadius;
        const color = continentColors[point.continent] || '#666';
        
        // Apply opacity based on layer
        if (isCityFiltered) {
          ctx.globalAlpha = 0.9; // High opacity for city filtered
        } else if (isSelected || isHovered) {
          ctx.globalAlpha = 1.0; // Full opacity for selected/hovered
        } else {
          ctx.globalAlpha = cityFilteredLocations.size > 0 ? 0.25 : 0.6; // Reduced opacity for regular points
        }
        
        // Draw point
        ctx.beginPath();
        ctx.arc(point.screenX, point.screenY, radius, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
        
        // Draw border with current theme-aware color
        if (isSelected || isHovered) {
          ctx.strokeStyle = isSelected ? '#ff6b6b' : '#ffd700';
          ctx.lineWidth = isSelected ? 2 : 1.5;
          ctx.stroke();
        } else if (isCityFiltered) {
          ctx.strokeStyle = '#4a90e2';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        } else {
          ctx.strokeStyle = currentOutlineColor; // Use current theme outline color
          ctx.lineWidth = 0.8;
          ctx.stroke();
        }
      });
    };
    
    // Render layers
    renderLayer(regularPoints);
    renderLayer(cityFilteredPoints, false, false, true);
    renderLayer(selectedPoints, true);
    renderLayer(hoveredPoint, false, true);
    
    ctx.restore();
  }, [dimensions, selectedLocations, cityFilteredLocations, continentColors, drawAxes, getOutlineColor]);

  // Center view on city tiles
  const centerOnCityTiles = (locationIds) => {
    if (!umapData || !scalesRef.current.xScale || locationIds.length === 0) return;

    const cityPoints = umapData.umap_points.filter(p => locationIds.includes(p.location_id));
    if (cityPoints.length === 0) return;

    const { xScale, yScale } = scalesRef.current;
    
    // Calculate bounding box of city points in UMAP space
    const xCoords = cityPoints.map(p => p.x);
    const yCoords = cityPoints.map(p => p.y);
    
    const minX = Math.min(...xCoords);
    const maxX = Math.max(...xCoords);
    const minY = Math.min(...yCoords);
    const maxY = Math.max(...yCoords);
    
    // Add padding
    const xPadding = (maxX - minX) * 0.2 || 0.1;
    const yPadding = (maxY - minY) * 0.2 || 0.1;
    
    const paddedMinX = minX - xPadding;
    const paddedMaxX = maxX + xPadding;
    const paddedMinY = minY - yPadding;
    const paddedMaxY = maxY + yPadding;
    
    // Calculate required scale to fit all points
    const plotWidth = dimensions.width - PLOT_MARGIN.left - PLOT_MARGIN.right;
    const plotHeight = dimensions.height - PLOT_MARGIN.top - PLOT_MARGIN.bottom;
    
    const scaleX = plotWidth / (xScale(paddedMaxX) - xScale(paddedMinX));
    const scaleY = plotHeight / (yScale(paddedMinY) - yScale(paddedMaxY)); // Note: Y is inverted
    
    const targetScale = Math.min(scaleX, scaleY, 5); // Cap at 5x zoom
    
    // Calculate center point - use actual container center, not viewport center
    const centerX = (paddedMinX + paddedMaxX) / 2;
    const centerY = (paddedMinY + paddedMaxY) / 2;
    
    // Use container dimensions instead of viewport dimensions
    const containerRect = containerRef.current?.getBoundingClientRect();
    const targetScreenX = containerRect ? containerRect.width / 2 : dimensions.width / 2;
    const targetScreenY = containerRect ? containerRect.height / 2 : dimensions.height / 2;
    
    const targetTransformX = targetScreenX - xScale(centerX) * targetScale;
    const targetTransformY = targetScreenY - yScale(centerY) * targetScale;

    // Smooth animation to new position
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      
      // Animation parameters
      const startTransformX = transformRef.current.x;
      const startTransformY = transformRef.current.y;
      const startScale = transformRef.current.k;
      const duration = 1000; // 1 second animation
      const startTime = performance.now();
      
      // Easing function for smooth animation
      const easeInOutCubic = (t) => {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
      };
      
      const animate = (currentTime) => {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easedProgress = easeInOutCubic(progress);
        
        // Interpolate between start and target positions
        transformRef.current.x = startTransformX + (targetTransformX - startTransformX) * easedProgress;
        transformRef.current.y = startTransformY + (targetTransformY - startTransformY) * easedProgress;
        transformRef.current.k = startScale + (targetScale - startScale) * easedProgress;
        
        // Render frame with current theme
        renderPoints(ctx, umapData.umap_points, transformRef.current);
        
        // Continue animation if not complete
        if (progress < 1) {
          requestAnimationFrame(animate);
        } else {
          // Rebuild quadtree after animation
          rebuildQuadtree();
        }
      };
      
      // Start animation
      requestAnimationFrame(animate);
    }
  };

  // Enhanced mouse event handlers with improved click/drag detection
  const handleMouseMove = useCallback((event) => {
    if (!umapData || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Check if mouse is in plot area
    const inPlotArea = x >= PLOT_MARGIN.left && x <= dimensions.width - PLOT_MARGIN.right &&
                      y >= PLOT_MARGIN.top && y <= dimensions.height - PLOT_MARGIN.bottom;
    
    // Update mouse position and calculate drag distance from initial mouse down position
    const currentPos = { x, y };
    
    // Handle dragging
    if (isDraggingRef.current && inPlotArea) {
      // Calculate total drag distance from initial mouse down position
      const totalDragDistance = Math.sqrt(
        Math.pow(currentPos.x - mouseDownPosRef.current.x, 2) + 
        Math.pow(currentPos.y - mouseDownPosRef.current.y, 2)
      );
      dragDistanceRef.current = totalDragDistance;
      
      // Calculate delta from last position for panning
      const deltaX = currentPos.x - lastMousePosRef.current.x;
      const deltaY = currentPos.y - lastMousePosRef.current.y;
      
      transformRef.current.x += deltaX;
      transformRef.current.y += deltaY;
      
      lastMousePosRef.current = currentPos;
      
      // Set grabbing cursor during drag
      canvas.style.cursor = 'grabbing';
      
      // Hide tooltip during dragging
      hideTooltip();
      
      // Clear any hovered point during dragging
      if (hoveredPointRef.current !== null) {
        hoveredPointRef.current = null;
      }
      
      // Throttled redraw during drag - ensure theme consistency
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      
      animationFrameRef.current = requestAnimationFrame(() => {
        const ctx = canvas.getContext('2d');
        renderPoints(ctx, umapData.umap_points, transformRef.current);
      });
      
      return; // Skip hover detection while dragging
    }
    
    // Store mouse position for potential drag start
    lastMousePosRef.current = currentPos;
    
    // Find closest point by checking distance to all points (only in plot area)
    // REDUCED SEARCH RADIUS for smaller points
    let closestPoint = null;
    let minDistance = 12; // Was 15px, now 12px search radius
    
    if (scalesRef.current.xScale && inPlotArea) {
      const { xScale, yScale } = scalesRef.current;
      const { k, x: tx, y: ty } = transformRef.current;
      
      umapData.umap_points.forEach(point => {
        const baseX = xScale(point.x);
        const baseY = yScale(point.y);
        const pointX = baseX * k + tx;
        const pointY = baseY * k + ty;
        
        const distance = Math.sqrt(Math.pow(x - pointX, 2) + Math.pow(y - pointY, 2));
        if (distance < minDistance) {
          minDistance = distance;
          closestPoint = point;
        }
      });
    }
    
    // Update cursor based on hover state and location
    if (closestPoint && inPlotArea) {
      canvas.style.cursor = 'pointer'; // Indicate clickable
    } else if (inPlotArea) {
      canvas.style.cursor = 'grab'; // Pan cursor in plot area
    } else {
      canvas.style.cursor = 'default'; // Default cursor outside plot
    }
    
    const newHoveredId = closestPoint ? closestPoint.location_id : null;
    
    if (hoveredPointRef.current !== newHoveredId) {
      hoveredPointRef.current = newHoveredId;
      
      // Throttled redraw with theme consistency
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      
      animationFrameRef.current = requestAnimationFrame(() => {
        const ctx = canvas.getContext('2d');
        renderPoints(ctx, umapData.umap_points, transformRef.current);
        
        // Show tooltip
        if (closestPoint) {
          showTooltip(closestPoint, x, y);
        } else {
          hideTooltip();
        }
      });
    }
  }, [umapData, renderPoints, dimensions]);

  const handleMouseDown = useCallback((event) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Only allow dragging in plot area
    const inPlotArea = x >= PLOT_MARGIN.left && x <= dimensions.width - PLOT_MARGIN.right &&
                      y >= PLOT_MARGIN.top && y <= dimensions.height - PLOT_MARGIN.bottom;
    
    if (inPlotArea) {
      isDraggingRef.current = true;
      dragStartTimeRef.current = performance.now();
      dragDistanceRef.current = 0;
      mouseDownPosRef.current = { x, y }; // Store initial mouse down position
      lastMousePosRef.current = { x, y };
    }
  }, [dimensions]);

  const handleMouseUp = useCallback(() => {
    const wasDragging = isDraggingRef.current;
    
    isDraggingRef.current = false;
    dragStartTimeRef.current = null;
    
    // Don't reset dragDistanceRef.current here - we need it for the click handler
    
    // Reset cursor - will be updated in next mousemove
    if (canvasRef.current) {
      canvasRef.current.style.cursor = 'grab';
    }
    
    // If we were dragging, hide tooltip and clear any hover state
    if (wasDragging) {
      hideTooltip();
      hoveredPointRef.current = null;
      
      // Re-render to clear any hover highlighting with current theme
      if (umapData && canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        renderPoints(ctx, umapData.umap_points, transformRef.current);
      }
    }
  }, [umapData, renderPoints]);

  const handleWheel = useCallback((event) => {
    event.preventDefault();
    
    if (!canvasRef.current) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    
    // Only zoom if mouse is in plot area
    const inPlotArea = mouseX >= PLOT_MARGIN.left && mouseX <= dimensions.width - PLOT_MARGIN.right &&
                      mouseY >= PLOT_MARGIN.top && mouseY <= dimensions.height - PLOT_MARGIN.bottom;
    
    if (!inPlotArea) return;
    
    // Reduced zoom sensitivity
    const scaleFactor = event.deltaY > 0 ? 0.95 : 1.05; // Was 0.9/1.1, now 0.95/1.05
    const newScale = Math.max(0.1, Math.min(10, transformRef.current.k * scaleFactor));
    
    // Zoom towards mouse position
    const scaleChange = newScale / transformRef.current.k;
    transformRef.current.x = mouseX - (mouseX - transformRef.current.x) * scaleChange;
    transformRef.current.y = mouseY - (mouseY - transformRef.current.y) * scaleChange;
    transformRef.current.k = newScale;
    
    // Throttled redraw with theme consistency
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    
    animationFrameRef.current = requestAnimationFrame(() => {
      const ctx = canvasRef.current.getContext('2d');
      renderPoints(ctx, umapData.umap_points, transformRef.current);
      
      // Rebuild quadtree after transform change
      rebuildQuadtree();
    });
  }, [umapData, renderPoints, rebuildQuadtree, dimensions]);

  const handleClick = useCallback((event) => {
    if (!umapData) return;
    
    // Enhanced drag detection with better logging
    const dragTime = dragStartTimeRef.current ? performance.now() - dragStartTimeRef.current : 0;
    const dragDistance = dragDistanceRef.current;
    
    // More conservative thresholds to prevent accidental clicks during pan
    const wasDragging = dragDistance > DRAG_THRESHOLD_PX || dragTime > DRAG_THRESHOLD_TIME;
    
    if (wasDragging) {
      // Reset drag distance after checking
      dragDistanceRef.current = 0;
      return;
    }
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Only handle clicks in plot area
    const inPlotArea = x >= PLOT_MARGIN.left && x <= dimensions.width - PLOT_MARGIN.right &&
                      y >= PLOT_MARGIN.top && y <= dimensions.height - PLOT_MARGIN.bottom;
    
    if (!inPlotArea) {
      dragDistanceRef.current = 0;
      return;
    }
    
    // Find closest point using same logic as hover with reduced radius
    let closestPoint = null;
    let minDistance = 12; // Reduced from 15px to match hover
    
    if (scalesRef.current.xScale) {
      const { xScale, yScale } = scalesRef.current;
      const { k, x: tx, y: ty } = transformRef.current;
      
      umapData.umap_points.forEach(point => {
        const baseX = xScale(point.x);
        const baseY = yScale(point.y);
        const pointX = baseX * k + tx;
        const pointY = baseY * k + ty;
        
        const distance = Math.sqrt(Math.pow(x - pointX, 2) + Math.pow(y - pointY, 2));
        if (distance < minDistance) {
          minDistance = distance;
          closestPoint = point;
        }
      });
    }
    
    if (closestPoint) {
      onLocationSelect(closestPoint.location_id);
      
      // Trigger map fly-to
      window.dispatchEvent(new CustomEvent('zoomToLocation', {
        detail: { 
          longitude: closestPoint.longitude, 
          latitude: closestPoint.latitude, 
          id: closestPoint.location_id 
        }
      }));
    }
    
    // Reset drag distance after processing click
    dragDistanceRef.current = 0;
  }, [umapData, onLocationSelect, dimensions]);

  // Tooltip functions
  const showTooltip = (point, x, y) => {
    let tooltip = document.getElementById('umap-tooltip');
    if (!tooltip) {
      tooltip = document.createElement('div');
      tooltip.id = 'umap-tooltip';
      tooltip.style.cssText = `
        position: absolute;
        background: var(--bg-overlay);
        color: var(--text-primary);
        padding: 8px;
        border-radius: 4px;
        pointer-events: none;
        font-size: 12px;
        z-index: 1000;
        opacity: 0;
        transition: opacity 0.2s;
        border: 1px solid var(--border-primary);
        box-shadow: var(--shadow-md);
      `;
      document.body.appendChild(tooltip);
    }
    
    tooltip.innerHTML = `
      <strong>${point.city}</strong><br/>
      ${point.country}<br/>
      <em>${point.continent}</em>
    `;
    
    const rect = canvasRef.current.getBoundingClientRect();
    tooltip.style.left = (rect.left + x + 10) + 'px';
    tooltip.style.top = (rect.top + y - 28) + 'px';
    tooltip.style.opacity = '0.9';
  };

  const hideTooltip = () => {
    const tooltip = document.getElementById('umap-tooltip');
    if (tooltip) {
      tooltip.style.opacity = '0';
    }
  };

  // Main rendering effect - Enhanced canvas sizing
  useEffect(() => {
    if (!umapData || !canvasRef.current || !dimensions.width) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Enhanced high DPI rendering with proper sizing
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = dimensions.width;
    const displayHeight = dimensions.height;
    
    // Set canvas size in memory (scaled for high DPI)
    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;
    
    // Set display size (CSS pixels)
    canvas.style.width = displayWidth + 'px';
    canvas.style.height = displayHeight + 'px';
    
    // Scale the context to match device pixel ratio
    ctx.scale(dpr, dpr);
    
    // Ensure canvas positioning
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    
    // Create scales with proper margins
    const plotWidth = dimensions.width - PLOT_MARGIN.left - PLOT_MARGIN.right;
    const plotHeight = dimensions.height - PLOT_MARGIN.top - PLOT_MARGIN.bottom;

    const xExtent = d3.extent(umapData.umap_points, d => d.x);
    const yExtent = d3.extent(umapData.umap_points, d => d.y);
    
    const xScale = d3.scaleLinear()
      .domain(xExtent)
      .range([PLOT_MARGIN.left, PLOT_MARGIN.left + plotWidth])
      .nice();

    const yScale = d3.scaleLinear()
      .domain(yExtent)
      .range([PLOT_MARGIN.top + plotHeight, PLOT_MARGIN.top])
      .nice();

    scalesRef.current = { xScale, yScale };

    // Build spatial index
    quadtreeRef.current = buildQuadtree(umapData.umap_points);

    // Initial render with current theme
    renderPoints(ctx, umapData.umap_points, transformRef.current);

    // Add event listeners
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mouseup', handleMouseUp);
    window.addEventListener('mouseup', handleMouseUp);     // Global to catch mouse up anywhere
    canvas.addEventListener('wheel', handleWheel);
    canvas.addEventListener('click', handleClick);
    canvas.addEventListener('mouseleave', () => {
      hideTooltip();
      hoveredPointRef.current = null;
    });

    return () => {
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mousedown', handleMouseDown);
      canvas.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('mouseup', handleMouseUp);
      canvas.removeEventListener('wheel', handleWheel);
      canvas.removeEventListener('click', handleClick);
      canvas.removeEventListener('mouseleave', hideTooltip);
      
      const tooltip = document.getElementById('umap-tooltip');
      if (tooltip) {
        tooltip.remove();
      }
    };
  }, [umapData, dimensions, selectedLocations, cityFilteredLocations, handleMouseMove, handleMouseDown, handleMouseUp, handleWheel, handleClick, buildQuadtree, renderPoints, rebuildQuadtree]);

  const zoomToPoint = (locationId) => {
    if (!umapData || !scalesRef.current.xScale) return;

    const targetPoint = umapData.umap_points.find(p => p.location_id === locationId);
    if (!targetPoint) return;

    const { xScale, yScale } = scalesRef.current;
    const targetX = xScale(targetPoint.x);
    const targetY = yScale(targetPoint.y);

    // Calculate target transform to center the point - use actual container center
    const containerRect = containerRef.current?.getBoundingClientRect();
    const centerX = containerRect ? containerRect.width / 2 : dimensions.width / 2;
    const centerY = containerRect ? containerRect.height / 2 : dimensions.height / 2;
    
    const currentScale = transformRef.current.k;
    const targetTransformX = centerX - targetX * currentScale;
    const targetTransformY = centerY - targetY * currentScale;

    // Smooth animation to new position
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      
      // Highlight the target point during animation
      hoveredPointRef.current = locationId;
      
      // Animation parameters
      const startTransformX = transformRef.current.x;
      const startTransformY = transformRef.current.y;
      const duration = 800; // 800ms animation
      const startTime = performance.now();
      
      // Easing function for smooth animation
      const easeInOutCubic = (t) => {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
      };
      
      const animate = (currentTime) => {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easedProgress = easeInOutCubic(progress);
        
        // Interpolate between start and target positions
        transformRef.current.x = startTransformX + (targetTransformX - startTransformX) * easedProgress;
        transformRef.current.y = startTransformY + (targetTransformY - startTransformY) * easedProgress;
        
        // Render frame with current theme
        renderPoints(ctx, umapData.umap_points, transformRef.current);
        
        // Continue animation if not complete
        if (progress < 1) {
          requestAnimationFrame(animate);
        } else {
          // Animation complete - reset highlight after a delay
          setTimeout(() => {
            if (hoveredPointRef.current === locationId) {
              hoveredPointRef.current = null;
              renderPoints(ctx, umapData.umap_points, transformRef.current);
            }
          }, 1000);
        }
      };
      
      // Start animation
      requestAnimationFrame(animate);
    }
  };

  const highlightPoint = (locationId) => {
    if (!canvasRef.current || !umapData) return;

    hoveredPointRef.current = locationId;
    
    const ctx = canvasRef.current.getContext('2d');
    renderPoints(ctx, umapData.umap_points, transformRef.current);

    // Reset highlight after delay
    setTimeout(() => {
      if (hoveredPointRef.current === locationId) {
        hoveredPointRef.current = null;
        renderPoints(ctx, umapData.umap_points, transformRef.current);
      }
    }, 2000);
  };

  if (loading) {
    return (
      <div className="umap-container">
        <div className="umap-plot" ref={containerRef}>
          <div className="loading-state">
            <div className="spinner"></div>
            <p>Loading UMAP embeddings...</p>
          </div>
        </div>
        <ColorLegend continentColors={continentColors} className="umap-legend" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="umap-container">
        <div className="umap-plot" ref={containerRef}>
          <div className="loading-state">
            <p>{error}</p>
            <button onClick={fetchUmapData} className="retry-btn">
              Retry
            </button>
          </div>
        </div>
        <ColorLegend continentColors={continentColors} className="umap-legend" />
      </div>
    );
  }

  return (
    <div className="umap-container">
      <div className="umap-plot" ref={containerRef}>
        <canvas 
          ref={canvasRef}
          style={{ 
            cursor: 'grab',
            display: 'block',
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%'
          }}
        />
        <ColorLegend continentColors={continentColors} className="umap-legend" />
      </div>
    </div>
  );
};

export default UMapView;