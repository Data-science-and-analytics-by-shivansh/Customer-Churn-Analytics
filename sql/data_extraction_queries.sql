-- =====================================================
-- CHURN PREDICTION DATA EXTRACTION QUERIES
-- =====================================================
-- These queries demonstrate SQL skills for data extraction
-- and feature engineering from a typical database schema
-- =====================================================

-- =====================================================
-- 1. MAIN CHURN DATASET EXTRACTION
-- =====================================================
-- Extract all customer data with churn labels
-- Assumes tables: customers, services, billing, churn_events

SELECT 
    c.customer_id,
    c.gender,
    c.senior_citizen,
    c.partner,
    c.dependents,
    c.tenure_months,
    
    -- Service information
    s.phone_service,
    s.multiple_lines,
    s.internet_service,
    s.online_security,
    s.online_backup,
    s.device_protection,
    s.tech_support,
    s.streaming_tv,
    s.streaming_movies,
    
    -- Contract and billing
    b.contract_type,
    b.paperless_billing,
    b.payment_method,
    b.monthly_charges,
    b.total_charges,
    
    -- Churn label
    CASE 
        WHEN ch.churn_date IS NOT NULL THEN 1 
        ELSE 0 
    END AS churn
    
FROM customers c
LEFT JOIN services s ON c.customer_id = s.customer_id
LEFT JOIN billing b ON c.customer_id = b.customer_id
LEFT JOIN churn_events ch ON c.customer_id = ch.customer_id
WHERE c.status IN ('Active', 'Churned')
  AND c.signup_date >= '2020-01-01';


-- =====================================================
-- 2. FEATURE ENGINEERING - CUSTOMER BEHAVIOR METRICS
-- =====================================================
-- Calculate advanced features for model training

WITH customer_metrics AS (
    SELECT 
        customer_id,
        
        -- Tenure features
        DATEDIFF(month, signup_date, COALESCE(churn_date, CURRENT_DATE)) AS tenure_months,
        DATEDIFF(day, signup_date, COALESCE(churn_date, CURRENT_DATE)) AS tenure_days,
        
        -- Engagement metrics
        COUNT(DISTINCT login_date) AS total_logins_last_90_days,
        AVG(session_duration_minutes) AS avg_session_duration,
        
        -- Support interaction
        COUNT(DISTINCT support_ticket_id) AS total_support_tickets,
        SUM(CASE WHEN ticket_status = 'Unresolved' THEN 1 ELSE 0 END) AS unresolved_tickets,
        
        -- Payment behavior
        SUM(CASE WHEN payment_status = 'Late' THEN 1 ELSE 0 END) AS late_payments,
        MAX(days_since_last_payment) AS days_since_last_payment
        
    FROM customer_activity
    WHERE activity_date >= DATEADD(day, -90, CURRENT_DATE)
    GROUP BY customer_id
)

SELECT 
    c.*,
    cm.tenure_months,
    cm.total_logins_last_90_days,
    cm.avg_session_duration,
    cm.total_support_tickets,
    cm.unresolved_tickets,
    cm.late_payments,
    cm.days_since_last_payment,
    
    -- Derived features
    CASE 
        WHEN cm.tenure_months <= 6 THEN 'New Customer'
        WHEN cm.tenure_months <= 24 THEN 'Regular Customer'
        ELSE 'Loyal Customer'
    END AS customer_segment,
    
    CASE 
        WHEN cm.total_logins_last_90_days < 5 THEN 'Low Engagement'
        WHEN cm.total_logins_last_90_days < 20 THEN 'Medium Engagement'
        ELSE 'High Engagement'
    END AS engagement_level
    
FROM customers c
LEFT JOIN customer_metrics cm ON c.customer_id = cm.customer_id;


-- =====================================================
-- 3. COHORT ANALYSIS - CHURN RATE BY SEGMENT
-- =====================================================
-- Analyze churn patterns across different customer segments

SELECT 
    contract_type,
    payment_method,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_pct,
    AVG(monthly_charges) AS avg_monthly_charges,
    AVG(tenure_months) AS avg_tenure_months
FROM churn_dataset
GROUP BY contract_type, payment_method
ORDER BY churn_rate_pct DESC;


-- =====================================================
-- 4. TIME-SERIES CHURN TREND
-- =====================================================
-- Track churn rate over time for monitoring

SELECT 
    DATE_TRUNC('month', churn_date) AS churn_month,
    COUNT(*) AS churned_customers,
    AVG(tenure_months) AS avg_tenure_at_churn,
    AVG(monthly_charges) AS avg_monthly_charges
FROM churn_events
WHERE churn_date >= DATEADD(month, -12, CURRENT_DATE)
GROUP BY DATE_TRUNC('month', churn_date)
ORDER BY churn_month;


-- =====================================================
-- 5. HIGH-RISK CUSTOMER IDENTIFICATION
-- =====================================================
-- Query to identify customers for proactive retention
-- (This would use model predictions in production)

WITH risk_factors AS (
    SELECT 
        customer_id,
        
        -- Risk signals
        CASE WHEN contract_type = 'Month-to-month' THEN 1 ELSE 0 END AS month_to_month_risk,
        CASE WHEN tenure_months <= 6 THEN 1 ELSE 0 END AS new_customer_risk,
        CASE WHEN payment_method = 'Electronic check' THEN 1 ELSE 0 END AS payment_risk,
        CASE WHEN monthly_charges > 70 THEN 1 ELSE 0 END AS high_price_risk,
        CASE WHEN tech_support = 'No' THEN 1 ELSE 0 END AS no_support_risk,
        
        -- Calculate total risk score
        (CASE WHEN contract_type = 'Month-to-month' THEN 1 ELSE 0 END +
         CASE WHEN tenure_months <= 6 THEN 1 ELSE 0 END +
         CASE WHEN payment_method = 'Electronic check' THEN 1 ELSE 0 END +
         CASE WHEN monthly_charges > 70 THEN 1 ELSE 0 END +
         CASE WHEN tech_support = 'No' THEN 1 ELSE 0 END) AS risk_score
         
    FROM churn_dataset
    WHERE churn = 0  -- Only active customers
)

SELECT 
    c.customer_id,
    c.gender,
    c.tenure_months,
    c.contract_type,
    c.monthly_charges,
    c.payment_method,
    rf.risk_score,
    CASE 
        WHEN rf.risk_score >= 4 THEN 'Critical Risk'
        WHEN rf.risk_score >= 3 THEN 'High Risk'
        WHEN rf.risk_score >= 2 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END AS risk_category
    
FROM customers c
INNER JOIN risk_factors rf ON c.customer_id = rf.customer_id
WHERE rf.risk_score >= 3  -- Focus on high-risk customers
ORDER BY rf.risk_score DESC, c.monthly_charges DESC
LIMIT 100;  -- Top 100 for immediate action


-- =====================================================
-- 6. RETENTION CAMPAIGN IMPACT ANALYSIS
-- =====================================================
-- Measure effectiveness of retention campaigns

SELECT 
    rc.campaign_id,
    rc.campaign_name,
    COUNT(DISTINCT rt.customer_id) AS customers_targeted,
    SUM(CASE WHEN c.churn = 1 THEN 1 ELSE 0 END) AS churned_after_campaign,
    ROUND(AVG(CASE WHEN c.churn = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct,
    
    -- Compare to baseline
    (SELECT AVG(churn) * 100 
     FROM churn_dataset 
     WHERE contract_type = 'Month-to-month') AS baseline_churn_rate_pct,
     
    -- Calculate lift
    ROUND(
        ((SELECT AVG(churn) * 100 FROM churn_dataset WHERE contract_type = 'Month-to-month') - 
         AVG(CASE WHEN c.churn = 1 THEN 1.0 ELSE 0.0 END) * 100) /
        (SELECT AVG(churn) * 100 FROM churn_dataset WHERE contract_type = 'Month-to-month') * 100,
        2
    ) AS retention_improvement_pct
    
FROM retention_campaigns rc
INNER JOIN retention_targets rt ON rc.campaign_id = rt.campaign_id
LEFT JOIN customers c ON rt.customer_id = c.customer_id
WHERE rc.campaign_start_date >= DATEADD(month, -6, CURRENT_DATE)
GROUP BY rc.campaign_id, rc.campaign_name
ORDER BY retention_improvement_pct DESC;


-- =====================================================
-- 7. DATA QUALITY CHECK
-- =====================================================
-- Validate data before model training

SELECT 
    'Total Records' AS metric,
    COUNT(*) AS value
FROM churn_dataset

UNION ALL

SELECT 
    'Missing Customer IDs' AS metric,
    SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) AS value
FROM churn_dataset

UNION ALL

SELECT 
    'Missing Tenure' AS metric,
    SUM(CASE WHEN tenure_months IS NULL THEN 1 ELSE 0 END) AS value
FROM churn_dataset

UNION ALL

SELECT 
    'Duplicate Customer IDs' AS metric,
    COUNT(*) - COUNT(DISTINCT customer_id) AS value
FROM churn_dataset

UNION ALL

SELECT 
    'Churn Rate' AS metric,
    ROUND(AVG(churn) * 100, 2) AS value
FROM churn_dataset

UNION ALL

SELECT 
    'Class Imbalance Ratio' AS metric,
    ROUND(
        CAST(SUM(CASE WHEN churn = 0 THEN 1 ELSE 0 END) AS FLOAT) / 
        NULLIF(SUM(CASE WHEN churn = 1 THEN 1 ELSE 0 END), 0),
        2
    ) AS value
FROM churn_dataset;