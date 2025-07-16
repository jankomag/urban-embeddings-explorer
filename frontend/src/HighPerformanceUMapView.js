import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as d3 from 'd3';

const HighPerformanceUMapView = ({ locations, selectedLocations, onLocationSelect }) => {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [umapData, setUmapData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  
  // Performance optimization refs
  const animationFrameRef = useRef(null);
  const quadtreeRef = useRef(null);
  const scalesRef = useRef({ xScale: null, yScale: null });
  const hoveredPointRef = useRef(null);
  const lastMousePosRef = useRef({ x: 0, y: 0 });
  const isDraggingRef = useRef(false);
  const transformRef = useRef({ k: 1, x: 0, y: 0 });

  const API_BASE_URL = process.env.NODE_ENV === 'production' 
    ? 'https://your-domain.com'
    : 'http://localhost:8000';

  // Continent color mapping - optimized for Canvas with opacity
  const continentColors = {
    'Africa': '#e74c3c',
    'Asia': '#3498db', 
    'Europe': '#2ecc71',
    'North America': '#f39c12',
    'South America': '#9b59b6',
    'Oceania': '#1abc9c',
    'Antarctica': '#95a5a6'
  };

  // Plot margins to ensure axes don't overlap with points
  const PLOT_MARGIN = { top: 30, right: 30, bottom: 60, left: 60 };

  // Throttled resize handler for performance
  const handleResize = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    
    animationFrameRef.current = requestAnimationFrame(() => {
      const container = containerRef.current;
      if (container) {
        const rect = container.getBoundingClientRect();
        const newDimensions = {
          width: Math.max(400, rect.width - 40),
          height: Math.max(300, rect.height - 40)
        };
        setDimensions(newDimensions);
      }
    });
  }, []);

  // Update dimensions on resize
  useEffect(() => {
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [handleResize]);

  // Fetch UMAP data
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

    window.addEventListener('highlightUmapPoint', handleHighlightPoint);
    window.addEventListener('zoomToUmapPoint', handleZoomToPoint);
    
    return () => {
      window.removeEventListener('highlightUmapPoint', handleHighlightPoint);
      window.removeEventListener('zoomToUmapPoint', handleZoomToPoint);
    };
  }, [umapData]);

  const fetchUmapData = async () => {
    setLoading(true);
    setError(null);

    try {
      console.log('Fetching UMAP data...');
      
      const response = await fetch(`${API_BASE_URL}/api/umap`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch UMAP data');
      }
      const data = await response.json();
      setUmapData(data);
      console.log(`UMAP data loaded: ${data.total_points} points`);
    } catch (err) {
      console.error('Error fetching UMAP data:', err);
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
    
    // Build quadtree with screen coordinates for accurate hit testing
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
    ctx.strokeStyle = '#dee2e6';
    ctx.fillStyle = '#666';
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

  // Optimized point rendering function with opacity and layering
  const renderPoints = useCallback((ctx, data, transform = { k: 1, x: 0, y: 0 }) => {
    if (!data || !scalesRef.current.xScale) return;
    
    const { xScale, yScale } = scalesRef.current;
    const { k, x: tx, y: ty } = transform;
    
    // Clear canvas
    ctx.clearRect(0, 0, dimensions.width, dimensions.height);
    
    // Draw axes first (they don't transform)
    drawAxes(ctx);
    
    // Render points with layering - selected points on top
    ctx.save();
    
    const plotWidth = dimensions.width - PLOT_MARGIN.left - PLOT_MARGIN.right;
    const plotHeight = dimensions.height - PLOT_MARGIN.top - PLOT_MARGIN.bottom;
    
    // Clip to plot area
    ctx.beginPath();
    ctx.rect(PLOT_MARGIN.left, PLOT_MARGIN.top, plotWidth, plotHeight);
    ctx.clip();
    
    // Separate points into layers
    const regularPoints = [];
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
      } else {
        regularPoints.push(pointData);
      }
    });
    
    // Render in layers: regular -> selected -> hovered
    const renderLayer = (points, isSelected = false, isHovered = false) => {
      points.forEach(point => {
        const baseRadius = isSelected ? 6 : 4;
        const radius = isHovered ? baseRadius + 2 : baseRadius;
        const color = continentColors[point.continent] || '#666';
        
        // Apply opacity to regular points
        ctx.globalAlpha = isSelected || isHovered ? 1.0 : 0.7;
        
        // Draw point
        ctx.beginPath();
        ctx.arc(point.screenX, point.screenY, radius, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
        
        // Draw border
        if (isSelected || isHovered) {
          ctx.strokeStyle = isSelected ? '#ff6b6b' : '#ffd700';
          ctx.lineWidth = isSelected ? 3 : 2;
          ctx.stroke();
        } else {
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      });
    };
    
    // Render layers
    renderLayer(regularPoints);
    renderLayer(selectedPoints, true);
    renderLayer(hoveredPoint, false, true);
    
    ctx.restore();
  }, [dimensions, selectedLocations, continentColors, drawAxes]);

  // Optimized mouse event handlers with throttling
  const handleMouseMove = useCallback((event) => {
    if (!umapData || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Check if mouse is in plot area
    const inPlotArea = x >= PLOT_MARGIN.left && x <= dimensions.width - PLOT_MARGIN.right &&
                      y >= PLOT_MARGIN.top && y <= dimensions.height - PLOT_MARGIN.bottom;
    
    // Update mouse position
    const currentPos = { x, y };
    
    // Handle dragging
    if (isDraggingRef.current && inPlotArea) {
      const deltaX = currentPos.x - lastMousePosRef.current.x;
      const deltaY = currentPos.y - lastMousePosRef.current.y;
      
      transformRef.current.x += deltaX;
      transformRef.current.y += deltaY;
      
      lastMousePosRef.current = currentPos;
      
      // Set grabbing cursor during drag
      canvas.style.cursor = 'grabbing';
      
      // Throttled redraw during drag
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
    let closestPoint = null;
    let minDistance = 15; // 15px search radius
    
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
      
      // Throttled redraw
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
      lastMousePosRef.current = { x, y };
    }
  }, [dimensions]);

  const handleMouseUp = useCallback(() => {
    isDraggingRef.current = false;
    // Reset cursor - will be updated in next mousemove
    if (canvasRef.current) {
      canvasRef.current.style.cursor = 'grab';
    }
  }, []);

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
    
    // Throttled redraw
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
    if (!umapData || isDraggingRef.current) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Only handle clicks in plot area
    const inPlotArea = x >= PLOT_MARGIN.left && x <= dimensions.width - PLOT_MARGIN.right &&
                      y >= PLOT_MARGIN.top && y <= dimensions.height - PLOT_MARGIN.bottom;
    
    if (!inPlotArea) return;
    
    // Find closest point using same logic as hover
    let closestPoint = null;
    let minDistance = 15; // 15px search radius
    
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
  }, [umapData, onLocationSelect, dimensions]);

  // Tooltip functions
  const showTooltip = (point, x, y) => {
    let tooltip = document.getElementById('umap-tooltip');
    if (!tooltip) {
      tooltip = document.createElement('div');
      tooltip.id = 'umap-tooltip';
      tooltip.style.cssText = `
        position: absolute;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 8px;
        border-radius: 4px;
        pointer-events: none;
        font-size: 12px;
        z-index: 1000;
        opacity: 0;
        transition: opacity 0.2s;
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

  // Main rendering effect
  useEffect(() => {
    if (!umapData || !canvasRef.current || !dimensions.width) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Set up high DPI rendering
    const dpr = window.devicePixelRatio || 1;
    canvas.width = dimensions.width * dpr;
    canvas.height = dimensions.height * dpr;
    canvas.style.width = dimensions.width + 'px';
    canvas.style.height = dimensions.height + 'px';
    ctx.scale(dpr, dpr);
    
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

    // Initial render
    renderPoints(ctx, umapData.umap_points, transformRef.current);

    // Add event listeners
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mouseup', handleMouseUp);
    window.addEventListener('mouseup', handleMouseUp);     // Global to catch mouse up anywhere
    canvas.addEventListener('wheel', handleWheel);
    canvas.addEventListener('click', handleClick);
    canvas.addEventListener('mouseleave', hideTooltip);

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
  }, [umapData, dimensions, selectedLocations, handleMouseMove, handleMouseDown, handleMouseUp, handleWheel, handleClick, buildQuadtree, renderPoints, rebuildQuadtree]);

  const zoomToPoint = (locationId) => {
    if (!umapData || !scalesRef.current.xScale) return;

    const targetPoint = umapData.umap_points.find(p => p.location_id === locationId);
    if (!targetPoint) return;

    const { xScale, yScale } = scalesRef.current;
    const targetX = xScale(targetPoint.x);
    const targetY = yScale(targetPoint.y);

    // Calculate target transform to center the point
    const centerX = dimensions.width / 2;
    const centerY = dimensions.height / 2;
    
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
        
        // Render frame
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
      <div className="umap-loading">
        <div className="spinner"></div>
        <p>Computing UMAP embeddings...</p>
        <small>Optimizing for high-performance rendering...</small>
      </div>
    );
  }

  if (error) {
    return (
      <div className="umap-error">
        <p>{error}</p>
        <button onClick={fetchUmapData} className="retry-btn">
          Retry
        </button>
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
            display: 'block'
          }}
        />
      </div>
      {umapData && (
        <div className="umap-stats">
          <small>
            {umapData.total_points.toLocaleString()} points • 
            Canvas rendering with persistent axes • 
            Click points to fly to location on map
          </small>
        </div>
      )}
    </div>
  );
};

export default HighPerformanceUMapView;