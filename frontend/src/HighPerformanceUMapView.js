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

  // Continent color mapping - optimized for Canvas
  const continentColors = {
    'Africa': '#e74c3c',
    'Asia': '#3498db', 
    'Europe': '#2ecc71',
    'North America': '#f39c12',
    'South America': '#9b59b6',
    'Oceania': '#1abc9c',
    'Antarctica': '#95a5a6'
  };

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

  // Listen for highlight events
  useEffect(() => {
    const handleHighlightPoint = (event) => {
      const { locationId } = event.detail;
      highlightPoint(locationId);
    };

    window.addEventListener('highlightUmapPoint', handleHighlightPoint);
    return () => window.removeEventListener('highlightUmapPoint', handleHighlightPoint);
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

  // Optimized point rendering function
  const renderPoints = useCallback((ctx, data, transform = { k: 1, x: 0, y: 0 }) => {
    if (!data || !scalesRef.current.xScale) return;
    
    const { xScale, yScale } = scalesRef.current;
    const { k, x: tx, y: ty } = transform;
    
    // Clear canvas
    ctx.clearRect(0, 0, dimensions.width, dimensions.height);
    
    // Render points without applying transform to context
    ctx.save();
    
    data.forEach(point => {
      const baseX = xScale(point.x);
      const baseY = yScale(point.y);
      
      // Apply transform manually for each point
      const x = baseX * k + tx;
      const y = baseY * k + ty;
      
      // Viewport culling - only render visible points
      if (x < -20 || x > dimensions.width + 20 || 
          y < -20 || y > dimensions.height + 20) {
        return;
      }
      
      const isSelected = selectedLocations.has(point.location_id);
      const isHovered = hoveredPointRef.current === point.location_id;
      
      // Fixed point size that doesn't scale with zoom
      const baseRadius = isSelected ? 6 : 4;
      const radius = isHovered ? baseRadius + 2 : baseRadius;
      const color = continentColors[point.continent] || '#666';
      
      // Draw point at screen coordinates
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
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
    
    ctx.restore();
  }, [dimensions, selectedLocations, continentColors]);

  // Optimized mouse event handlers with throttling
  const handleMouseMove = useCallback((event) => {
    if (!umapData || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Update mouse position
    const currentPos = { x, y };
    
    // Handle dragging
    if (isDraggingRef.current) {
      const deltaX = currentPos.x - lastMousePosRef.current.x;
      const deltaY = currentPos.y - lastMousePosRef.current.y;
      
      transformRef.current.x += deltaX;
      transformRef.current.y += deltaY;
      
      lastMousePosRef.current = currentPos;
      
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
    
    // Find closest point by checking distance to all points (more reliable than quadtree for transformed coordinates)
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
  }, [umapData, renderPoints]);

  const handleMouseDown = useCallback((event) => {
    isDraggingRef.current = true;
    const rect = canvasRef.current.getBoundingClientRect();
    lastMousePosRef.current = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    };
    
    // Change cursor to indicate dragging
    if (canvasRef.current) {
      canvasRef.current.style.cursor = 'grabbing';
    }
  }, []);

  const handleMouseUp = useCallback(() => {
    isDraggingRef.current = false;
    
    // Reset cursor
    if (canvasRef.current) {
      canvasRef.current.style.cursor = 'grab';
    }
  }, []);

  const handleMouseDrag = useCallback((event) => {
    // This function is now handled within handleMouseMove
    // Keeping it for consistency but it's not used
  }, []);

  const handleWheel = useCallback((event) => {
    event.preventDefault();
    
    if (!canvasRef.current) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    
    // Zoom scaling
    const scaleFactor = event.deltaY > 0 ? 0.9 : 1.1;
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
  }, [umapData, renderPoints, rebuildQuadtree]);

  const handleClick = useCallback((event) => {
    if (!umapData || isDraggingRef.current) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
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
    }
  }, [umapData, onLocationSelect]);

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
    
    // Create scales
    const margin = { top: 20, right: 20, bottom: 60, left: 60 };
    const plotWidth = dimensions.width - margin.left - margin.right;
    const plotHeight = dimensions.height - margin.top - margin.bottom;

    const xExtent = d3.extent(umapData.umap_points, d => d.x);
    const yExtent = d3.extent(umapData.umap_points, d => d.y);
    
    const xScale = d3.scaleLinear()
      .domain(xExtent)
      .range([margin.left, margin.left + plotWidth])
      .nice();

    const yScale = d3.scaleLinear()
      .domain(yExtent)
      .range([margin.top + plotHeight, margin.top])
      .nice();

    scalesRef.current = { xScale, yScale };

    // Build spatial index
    quadtreeRef.current = buildQuadtree(umapData.umap_points);

    // Initial render
    renderPoints(ctx, umapData.umap_points, transformRef.current);

    // Draw axes and labels (using Canvas for consistency)
    ctx.save();
    ctx.strokeStyle = '#dee2e6';
    ctx.lineWidth = 1;
    
    // Draw axes
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top + plotHeight);
    ctx.lineTo(margin.left + plotWidth, margin.top + plotHeight);
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + plotHeight);
    ctx.stroke();
    
    // Draw axis labels
    ctx.fillStyle = '#666';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('UMAP Dimension 1', margin.left + plotWidth / 2, dimensions.height - 15);
    
    ctx.save();
    ctx.translate(15, margin.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('UMAP Dimension 2', 0, 0);
    ctx.restore();
    
    ctx.restore();

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
      <div className="umap-header">
        <h3>High-Performance UMAP Visualization</h3>
        <p>
          Canvas-accelerated TerraMind satellite embeddings • 
          Hardware-optimized for {umapData?.total_points?.toLocaleString()} points • 
          Drag to pan, scroll to zoom
        </p>
        {umapData && (
          <div className="umap-stats">
            <small>
              {umapData.total_points.toLocaleString()} points • 
              Canvas rendering with spatial indexing • 
              60fps interactions
            </small>
          </div>
        )}
      </div>
      <div className="umap-plot" ref={containerRef}>
        <canvas 
          ref={canvasRef}
          style={{ 
            cursor: 'grab',
            display: 'block'
          }}
        />
      </div>
    </div>
  );
};

export default HighPerformanceUMapView;